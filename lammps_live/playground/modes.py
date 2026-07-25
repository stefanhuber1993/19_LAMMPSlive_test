"""Interaction modes: how (or whether) the user drives the simulation.

This is the split that lets one playground be both a game and an experiment.
Previously "has a puller" was baked into each system class -- MDSystem declared 7
puller-shaped abstract methods, and the self-assembly system had to stub five of
them and invent a `SliderSpec("Puller damping (unused)", ...)` to satisfy a
required field. Now:

    GameMode  one controlled particle driven by joystick/mouse/keyboard force,
              held on a control plane inside a leash, with haptics and the
              force-field reaction force recovered for force feedback.
    SimMode   no control particle; Play / Pause / Reset, and Reset rebuilds from
              a fresh random state.

Because the mode is a separate object, `--mode game` and `--mode sim` both work
on any playground -- so you can watch a structure self-assemble and then grab a
particle and probe it, which was not possible at any price before.
"""
import math

import numpy as np


def select_controlled(positions, box, selector):
    """Index of the particle the user controls.

    Selectors:
      "first"           -- the first particle (a scenario that puts its centre
                           site first, like the hex patch, gets that centre)
      "nearest_center"  -- nearest the box centre, measured in-plane
      "nearest:x,y,z"   -- nearest a given point
    """
    if not len(positions):
        return None
    if selector in (None, "none"):
        return None
    if selector == "first":
        return 0
    if selector == "last":
        # A deposition scenario creates its slab first and the free atom last, so
        # the highest id is the one meant to be driven.
        return len(positions) - 1
    if selector == "nearest_center":
        target = np.array(box.center if box is not None else (0.0, 0.0, 0.0))
        return int(np.argmin(np.linalg.norm(positions[:, :2] - target[:2], axis=1)))
    if isinstance(selector, str) and selector.startswith("nearest:"):
        target = np.array([float(t) for t in selector[len("nearest:"):].split(",")])
        return int(np.argmin(np.linalg.norm(positions - target, axis=1)))
    if isinstance(selector, (int, np.integer)):
        return int(selector)
    raise ValueError(f"unrecognized control-atom selector {selector!r}")


_AXES = {"x": 0, "y": 1, "z": 2}


def plane_axes(plane):
    """(u_axis, v_axis, pinned_axis) indices for a two-letter plane name.

    The control plane is the plane the two input axes slide the particle in;
    the third axis is pinned so a 2-axis stick fully determines its 3D position.
    "xz" -- the shipped choice -- puts the plane perpendicular to a membrane
    lying in xy and facing the camera, so input x is screen-horizontal and input
    y is out-of-plane.
    """
    if len(plane) != 2 or any(c not in _AXES for c in plane):
        raise ValueError(f"plane must be two of x/y/z, got {plane!r}")
    u, v = _AXES[plane[0]], _AXES[plane[1]]
    pin = ({0, 1, 2} - {u, v}).pop()
    return u, v, pin


class Mode:
    """Base mode. `attach` gives it the runtime it drives."""

    playback_controls = False
    needs_control_particle = False

    def attach(self, runtime):
        self.runtime = runtime

    # --- deck contributions ---------------------------------------------------

    def group_commands(self):
        return []

    def thermostat_group(self):
        """The group the Langevin bath acts on."""
        return "all"

    def control_commands(self, params):
        return []

    # --- per-frame ------------------------------------------------------------

    def after_step(self, dt):
        pass

    # --- the app-facing readouts ---------------------------------------------

    def set_input_force(self, fx, fy):
        pass

    def set_damping(self, gamma):
        pass

    def steer_orientation(self, rate, dt):
        pass

    def controlled_index(self):
        return None

    def puller_state(self):
        return None, None

    def interaction_force(self):
        return np.array([0.0, 0.0])

    def torque_signals(self):
        return None

    def control_grid(self):
        return None


class SimMode(Mode):
    """Watch it run: Play / Pause / Reset, no interactive particle.

    Every puller-shaped readout returns a neutral value, which the app already
    handles -- but now that is one small class rather than five stub methods
    copied into a system that has no puller.
    """

    playback_controls = True

    def thermostat_group(self):
        return "all"


class GameMode(Mode):
    """Drive one particle with an input device, and feel the force field push
    back.

    The particle is confined to a control plane and a rectangular leash, and
    capped below a runaway speed: without this a sustained max pull would
    accelerate it (undamped by the thermostat, since it is deliberately excluded
    from the bath) straight out of the box. The leash is drawn in the scene as a
    net, whose extents ARE these limits, so the net marks exactly where the
    particle can be dragged.
    """

    needs_control_particle = True

    def __init__(self, control):
        self.control = control
        self.u_axis, self.v_axis, self.pin_axis = plane_axes(control.plane)
        self._input_u = 0.0
        self._input_v = 0.0
        self._yaw = 0.0
        self._damping = control.damping_default
        self._pin_value = 0.0
        self._has_group_force = False

    # --- deck ----------------------------------------------------------------

    def group_commands(self):
        cid = self.runtime.controlled_id
        if cid is None:
            # No particle could be selected (an empty scenario, or atom="none").
            # Degrade to thermostatting everything rather than emitting a broken
            # group command; every readout below already handles a missing
            # controlled particle.
            return []
        return [
            f"group controlled id {cid}",
            # The bath thermostats everything EXCEPT the controlled particle.
            # Never `all`: folding the controlled particle into the Langevin bath
            # gives its director thermal rotational noise that never goes away,
            # and silently breaks the reaction-force recovery below (this was a
            # real bug once -- see interaction_force).
            "group bath subtract all controlled",
        ]

    def thermostat_group(self):
        return "bath" if self.runtime.controlled_id is not None else "all"

    def control_commands(self, params):
        if self.runtime.controlled_id is None:
            return []
        cmds = ["fix drive controlled addforce 0.0 0.0 0.0",
                f"fix damp controlled viscous {self._damping}"]
        if self.control.displacement_cap:
            # An unconfined particle needs a per-step displacement cap to survive
            # a hard contact impact instead of tunnelling through the lattice.
            cmds.append(f"fix integ_controlled controlled nve/limit "
                        f"{self.control.displacement_cap}")
        if self.control.confine:
            # Zero the out-of-plane force every step. Defined last so it also
            # cancels any out-of-plane interaction force.
            pin = ["NULL", "NULL", "NULL"]
            pin[self.pin_axis] = "0.0"
            cmds.append(f"fix plane controlled setforce {' '.join(pin)}")
        if self.runtime.force_field.supports_single:
            # A pair style with single() can report the force between two groups
            # directly, so the reconstruction below is unnecessary -- and exact
            # rather than dependent on knowing every fix acting on the particle.
            cmds.append("compute pairforce controlled group/group crystal")
            self._has_group_force = True
        return cmds

    def on_built(self):
        """Record the controlled particle's home coordinate on the pinned axis --
        the plane it is held in, and the origin of the drawn net."""
        pos = self.runtime.controlled_position()
        if pos is not None:
            self._pin_value = float(pos[self.pin_axis])
        self.constrain()

    # --- inputs --------------------------------------------------------------

    def set_input_force(self, fx, fy):
        """Input axis 1 -> the plane's u axis, axis 2 -> its v axis."""
        self._input_u = fx
        self._input_v = fy
        if self.runtime.controlled_id is None:
            return
        f = [0.0, 0.0, 0.0]
        f[self.u_axis] = fx
        f[self.v_axis] = fy
        self.runtime.lmp.command(
            f"fix drive controlled addforce {f[0]} {f[1]} {f[2]}"
        )

    def set_damping(self, gamma):
        lo, hi = self.control.damping_range
        gamma = max(lo, min(hi, gamma))
        if gamma == self._damping:
            return
        self._damping = gamma
        if self.runtime.controlled_id is not None:
            self.runtime.lmp.command(f"fix damp controlled viscous {gamma}")

    def steer_orientation(self, rate, dt):
        # Sign flipped so the twist turns the director the way the hand expects
        # on screen. Applied in the next constrain().
        self._yaw = -rate

    # --- per-frame constraint ------------------------------------------------

    def after_step(self, dt):
        self.constrain()

    def constrain(self):
        """Hold the controlled particle on its plane, inside the leash, below the
        speed cap, and drive its director.

        The director spring-back the user feels is genuine force-field physics
        (the tilt term, integrated by LAMMPS). Only three things are added here:
        the rotation is constrained to the control plane, the yaw command enters
        as an angular-momentum kick (a torque is dL/dt), and a rotational drag
        stands in for the controlled particle's share of the implicit solvent's
        rotational friction -- it sits outside the Langevin bath, so without this
        an undamped director would oscillate forever.
        """
        if not self.control.confine:
            return
        ic = self.runtime.controlled_local()
        if ic is None:
            return
        lmp = self.runtime.lmp
        x = lmp.numpy.extract_atom("x")
        v = lmp.numpy.extract_atom("v")

        # Exact plane constraint (belt-and-braces with the setforce fix).
        x[ic][self.pin_axis] = self._pin_value
        v[ic][self.pin_axis] = 0.0
        for axis, (lo, hi) in ((self.u_axis, self.control.u_range),
                               (self.v_axis, self.control.v_range)):
            if x[ic][axis] < lo:
                x[ic][axis] = lo
                if v[ic][axis] < 0.0:
                    v[ic][axis] = 0.0
            elif x[ic][axis] > hi:
                x[ic][axis] = hi
                if v[ic][axis] > 0.0:
                    v[ic][axis] = 0.0
        speed = math.sqrt(v[ic][0] ** 2 + v[ic][1] ** 2 + v[ic][2] ** 2)
        cap = self.control.speed_cap
        if speed > cap:
            s = cap / speed
            v[ic][0] *= s
            v[ic][1] *= s
            v[ic][2] *= s

        if not self.runtime.has_directors:
            return
        mu = lmp.numpy.extract_atom("mu")
        omega = lmp.numpy.extract_atom("omega")
        # Spin only about the pinned axis, so the director's swing stays in the
        # control plane.
        omega[ic][self.u_axis] = 0.0
        omega[ic][self.v_axis] = 0.0
        omega[ic][self.pin_axis] = (omega[ic][self.pin_axis] * self.control.rot_damp
                                    + self._yaw * self.control.yaw_torque)
        # Kill any out-of-plane director drift.
        nu, nv = mu[ic][self.u_axis], mu[ic][self.v_axis]
        m = math.hypot(nu, nv)
        if m > 1e-9:
            mu[ic][self.u_axis] = nu / m
            mu[ic][self.pin_axis] = 0.0
            mu[ic][self.v_axis] = nv / m

    # --- readouts ------------------------------------------------------------

    def controlled_index(self):
        return self.runtime.controlled_index

    def puller_state(self):
        """(position, velocity) projected onto the control plane, so the app's
        2D haptics pipeline works unchanged on a 3D scene."""
        ic = self.runtime.controlled_local()
        if ic is None:
            return None, None
        n = self.runtime.natoms
        x = self.runtime.lmp.numpy.extract_atom("x")[:n]
        v = self.runtime.lmp.numpy.extract_atom("v")[:n]
        return (np.array([x[ic][self.u_axis], x[ic][self.v_axis]]),
                np.array([v[ic][self.u_axis], v[ic][self.v_axis]]))

    def interaction_force(self):
        """The force field's reaction force on the controlled particle, projected
        onto the control plane.

        Recovered as total force minus the two forces we apply ourselves: the
        input drive (addforce) and the viscous damping (-gamma*v). The setforce on
        the pinned axis is irrelevant because only the two in-plane components are
        read.

        This substitutes for `compute group/group`, which a pair style without a
        single() method cannot support -- and MesoMem's has none. It is therefore
        only correct while (a) exactly these two fixes act on the particle in the
        plane, (b) _input_u/_input_v mirror the last addforce issued, and (c) the
        particle is excluded from the thermostat. All three are enforced above,
        and a regression test pins the arithmetic.
        """
        if self._has_group_force:
            vec = self.runtime.lmp.extract_compute("pairforce", 0, 1)
            return np.array([vec[self.u_axis], vec[self.v_axis]])
        ic = self.runtime.controlled_local()
        if ic is None:
            return np.array([0.0, 0.0])
        n = self.runtime.natoms
        f = self.runtime.lmp.numpy.extract_atom("f")[:n]
        v = self.runtime.lmp.numpy.extract_atom("v")[:n]
        g = self._damping
        return np.array([
            f[ic][self.u_axis] - self._input_u + g * v[ic][self.u_axis],
            f[ic][self.v_axis] - self._input_v + g * v[ic][self.v_axis],
        ])

    def torque_signals(self):
        """(applied, reaction) torques about the control plane's normal,
        normalized to [-1, 1] for the circular torque arrows.

        `applied` is the user's yaw command, which is already the per-frame
        angular kick in [-1, 1]. `reaction` is the force field's restoring torque,
        read straight off the pair style's per-atom torque -- the component about
        the pinned axis is the part that rotates the director within the plane.
        """
        ic = self.runtime.controlled_local()
        if ic is None or not self.runtime.has_directors:
            return None
        applied = max(-1.0, min(1.0, self._yaw))
        tau = self.runtime.lmp.numpy.extract_atom("torque")
        reaction = 0.0
        if tau is not None:
            raw = float(tau[:self.runtime.natoms][ic][self.pin_axis])
            reaction = max(-1.0, min(1.0, raw / self.control.reaction_torque_max))
        return applied, reaction

    def control_grid(self):
        """The net: the control plane's basis and extents. Drawn exactly at the
        leash limits, so it marks precisely where the particle can be dragged.
        None for an unconfined particle -- there is no boundary to draw."""
        if not self.control.confine:
            return None
        origin = [0.0, 0.0, 0.0]
        origin[self.pin_axis] = self._pin_value
        u = [0.0, 0.0, 0.0]
        u[self.u_axis] = 1.0
        v = [0.0, 0.0, 0.0]
        v[self.v_axis] = 1.0
        return dict(origin=tuple(origin), u_axis=tuple(u), v_axis=tuple(v),
                    u_range=self.control.u_range, v_range=self.control.v_range,
                    step=self.control.grid_step)
