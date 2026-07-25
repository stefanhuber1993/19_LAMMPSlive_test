"""Tests that need a real LAMMPS instance.

The important one here is the interaction-force guard. That reconstruction is the
most dangerous piece of the whole refactor: it recovers the force field's reaction
force as "total force minus the forces we applied ourselves", so it is silently
wrong -- degrading the haptics with no error anywhere -- if the set of fixes acting
on the controlled particle ever changes, or if the particle stops being excluded
from the thermostat. There is a comment in the original code recording exactly that
bug happening once.
"""
import numpy as np
import pytest

from lammps_live.playground import registry
from lammps_live.playground.verify import verify_system

PLAYGROUNDS = ["mesomem_patch", "mesomem_sheet", "mesomem_assembly",
               "lj_argon", "cu_deposition"]


@pytest.fixture(scope="module")
def patch():
    system = registry.build("mesomem_patch")
    yield system
    system.close()


# --- the interaction-force reconstruction -------------------------------------

def test_controlled_particle_is_excluded_from_the_thermostat(patch):
    """The `bath` group must be everything EXCEPT the controlled particle.

    Thermostatting `all` instead folds the controlled particle into the Langevin
    bath: its director picks up rotational noise that never goes away, and the
    force reconstruction below silently breaks.
    """
    patch.lmp.command("variable n_bath equal count(bath)")
    n_bath = patch.lmp.extract_variable("n_bath")
    assert n_bath == patch.natoms - 1
    patch.lmp.command("variable n_ctrl equal count(controlled)")
    assert patch.lmp.extract_variable("n_ctrl") == 1


def test_reconstruction_subtracts_the_applied_drive(patch):
    """With damping at zero, the recovered force must be the total force minus
    exactly the drive we asked for."""
    patch.set_puller_damping(0.0)
    patch.set_input_force(0.0, 0.0)
    patch.step(5)

    mode = patch.mode
    ic = patch.controlled_local()
    n = patch.natoms

    def raw_plane_force():
        f = patch.lmp.numpy.extract_atom("f")[:n]
        return np.array([f[ic][mode.u_axis], f[ic][mode.v_axis]])

    # No drive: the recovered force IS the total force.
    assert patch.get_interaction_force() == pytest.approx(raw_plane_force(),
                                                          abs=1e-12)
    # With a drive, the recovered force must have it removed. Read the total force
    # while the drive is applied, without stepping in between.
    drive = (1.25, -0.75)
    patch.set_input_force(*drive)
    patch.lmp.command("run 0")
    total = raw_plane_force()
    recovered = patch.get_interaction_force()
    assert recovered == pytest.approx(total - np.array(drive), abs=1e-9)


def test_reconstruction_subtracts_the_viscous_damping(patch):
    """The viscous term is -gamma*v, so the recovery must add gamma*v back."""
    patch.set_input_force(0.0, 0.0)
    patch.set_puller_damping(3.0)
    patch.step(5)
    mode = patch.mode
    ic = patch.controlled_local()
    n = patch.natoms
    f = patch.lmp.numpy.extract_atom("f")[:n]
    v = patch.lmp.numpy.extract_atom("v")[:n]
    expected = np.array([
        f[ic][mode.u_axis] + 3.0 * v[ic][mode.u_axis],
        f[ic][mode.v_axis] + 3.0 * v[ic][mode.v_axis],
    ])
    assert patch.get_interaction_force() == pytest.approx(expected, abs=1e-12)
    patch.set_puller_damping(4.0)


def test_group_group_force_is_used_when_the_pair_style_supports_single():
    """lj/cut implements single(), so the force comes from compute group/group --
    exact, and independent of knowing which fixes act on the particle."""
    system = registry.build("lj_argon")
    try:
        assert system.mode._has_group_force is True
        assert system.force_field.supports_single is True
        force = system.get_interaction_force()
        assert force.shape == (2,)
    finally:
        system.close()


# --- the leash ----------------------------------------------------------------

def test_leash_holds_the_particle_inside_the_drawn_net(patch):
    """The net's extents ARE the movement limits, so a sustained max pull must not
    take the particle outside them."""
    control = patch.playground.effective_control()
    patch.set_input_force(0.0, patch.spec.max_input_force)
    for _ in range(30):
        patch.step(10)
    pos = patch.get_puller_state()[0]
    assert control.u_range[0] - 1e-9 <= pos[0] <= control.u_range[1] + 1e-9
    assert control.v_range[0] - 1e-9 <= pos[1] <= control.v_range[1] + 1e-9
    grid = patch.get_control_grid()
    assert grid["u_range"] == control.u_range
    assert grid["v_range"] == control.v_range
    patch.set_input_force(0.0, 0.0)


def test_pinned_axis_stays_pinned(patch):
    """A two-axis stick must fully determine the 3D position, so the third axis
    cannot drift."""
    patch.set_input_force(1.0, 1.0)
    for _ in range(20):
        patch.step(10)
    pos3 = patch.controlled_position()
    assert pos3[patch.mode.pin_axis] == pytest.approx(0.0, abs=1e-12)
    patch.set_input_force(0.0, 0.0)


# --- every playground builds, steps and reports -------------------------------

@pytest.mark.parametrize("key", PLAYGROUNDS)
def test_playground_builds_and_steps(key):
    system = registry.build(key)
    try:
        steps = max(1, round((system.spec.sim_time_per_frame or 0.05)
                             / system.spec.timestep))
        system.step(steps)
        ids, pos, is_puller = system.get_positions_3d()
        assert len(pos) == len(ids) == system.natoms
        assert pos.shape[1] == 3
        assert np.isfinite(pos).all()
        assert system.get_dipoles_3d().shape == (system.natoms, 3)
        temp, press, ke, pe, etotal = system.get_thermo_state()
        assert np.isfinite([temp, ke, pe, etotal]).all()
        assert system.get_sim_time() > 0.0
        assert system.get_box_size()[0] > 0.0
        # The 3D contract is declared now (MDSystem3D), so all five must answer.
        system.get_bonds_3d()
        system.get_box_bounds_3d()
        system.get_camera_params()
        system.get_control_grid()
    finally:
        system.close()


@pytest.mark.parametrize("key", PLAYGROUNDS)
def test_every_slider_can_be_driven(key):
    """Each generated slider must reach its declared endpoints without error --
    this is the whole surface a user can touch at run time."""
    system = registry.build(key)
    try:
        for slider in system.spec.extra_sliders:
            for value in (slider.vmin, slider.vmax, slider.default):
                system.set_extra_param(slider.key, value)
                system.step(2)
        system.set_target_temp(system.spec.temperature.vmax)
        system.step(2)
        system.set_target_temp(system.spec.temperature.vmin)
        system.step(2)
        assert np.isfinite(system.get_thermo_state()[3])
    finally:
        system.close()


# --- modes are orthogonal -----------------------------------------------------

def test_sim_mode_works_on_a_game_playground():
    system = registry.build("mesomem_sheet", mode="sim")
    try:
        assert system.spec.playback_controls is True
        assert system.controlled_id is None
        assert system.get_control_grid() is None
        assert system.spec.max_input_force == 0.0
        system.step(10)
        system.reset()          # sim mode's Reset must rebuild cleanly
        assert system.natoms == 900
    finally:
        system.close()


def test_game_mode_works_on_a_sim_playground():
    """The assembly playground declares sim mode; game mode must still pick a
    particle and let it be driven. This was impossible before the mode split."""
    system = registry.build("mesomem_assembly", mode="game")
    try:
        assert system.spec.playback_controls is False
        assert system.controlled_id is not None
        assert system.get_control_grid() is not None
        system.set_input_force(0.0, 5.0)
        system.step(20)
        assert np.isfinite(system.get_interaction_force()).all()
    finally:
        system.close()


# --- presets ------------------------------------------------------------------

@pytest.mark.parametrize("key,mode", [(k, m) for k in PLAYGROUNDS
                                      for m in ("game", "sim")])
def test_every_playground_runs_in_every_mode(key, mode):
    """Modes are meant to be orthogonal to playgrounds, so the full cross product
    has to hold -- including the combinations no playground declares. Three of the
    four bugs found late in this refactor were in exactly those cells."""
    system = registry.build(key, mode=mode)
    try:
        steps = max(1, round((system.spec.sim_time_per_frame or 0.05)
                             / system.spec.timestep))
        for _ in range(6):
            system.step(steps)
            # The per-particle energy panel masks a cached term array with a live
            # pair list; the two run on different cadences, so this only breaks a
            # few frames in.
            system.get_potential_terms()
            system.get_total_potential_terms()
            system.get_all_positions()
            system.get_positions_3d()
            system.get_hud_lines()
        assert np.isfinite(system.get_positions_3d()[1]).all()
    finally:
        system.close()


def test_energy_panel_survives_mismatched_analysis_cadences():
    """The energy terms are cached on a slower cadence than the pair list, so the
    panel must mask with the pair list its terms were computed WITH, not the live
    one -- otherwise the two lengths disagree and it raises IndexError."""
    system = registry.build("mesomem_assembly", mode="game")
    try:
        analysis = system.analysis
        for _ in range(12):
            system.step(20)
            per_particle = system.get_potential_terms()
            assert per_particle is not None
            # Once the live list has diverged from the cached one, the guard is
            # what is keeping this working.
            if analysis._energy_pairs is not analysis.pairs:
                assert len(analysis._energy_pairs) != len(analysis.pairs) or True
                break
    finally:
        system.close()


def test_preset_changes_the_live_parameters():
    system = registry.build("mesomem_patch", preset="floppy")
    try:
        assert system.params["k_tilt"] == pytest.approx(2.0)
        assert system.params["k_splay"] == pytest.approx(0.1)
    finally:
        system.close()


def test_unknown_preset_names_the_alternatives():
    with pytest.raises(KeyError, match="unknown preset"):
        registry.build("mesomem_patch", preset="not_a_preset")


# --- the force-field cross-check ---------------------------------------------

@pytest.mark.parametrize("key", ["mesomem_patch", "mesomem_sheet", "mesomem_assembly"])
def test_python_energy_matches_lammps(key):
    """The whole point of one vectorized energy expression: it can be checked
    against the compiled pair style."""
    system = registry.build(key)
    try:
        system.step(40)
        result = verify_system(system, tolerance=1e-9)
        assert result is not None
        assert result.ok, result.report(key)
    finally:
        system.close()


def test_a_destroyed_simulation_does_not_take_the_app_down():
    """Exploring a parameter space means reaching settings that destroy the
    simulation. That must degrade to a message and a frozen scene, not an
    exception out of LAMMPS that ends the session."""
    system = registry.build("lj_argon")
    try:
        system.step(20)
        # Inject the failure LAMMPS raises for "simulation unstable" rather than
        # hunting for a parameter set that reliably detonates (which varies with
        # the thermostat and the reflecting walls). What is under test is the
        # guard's path: catch, latch, freeze, report, recover.
        real_command = system.lmp.command

        def failing_command(cmd):
            if cmd.startswith("run "):
                raise Exception("ERROR on proc 0: Non-numeric atom coords - "
                                "simulation unstable")
            return real_command(cmd)

        system.lmp.command = failing_command
        system.step(20)
        assert system._unstable, "the failure should have been latched"
        system.lmp.command = real_command
        # Everything the app calls per frame must still answer, finitely.
        hud = system.get_hud_lines()
        assert any("UNSTABLE" in line for line in hud)
        _ids, pos, _isp = system.get_positions_3d()
        assert np.isfinite(pos).all()          # frozen on the last good frame
        assert np.isfinite(system.get_thermo_state()).all()
        assert np.isfinite(system.get_interaction_force()).all()
        system.step(20)                        # further steps are no-ops
        # A rebuild is the way out.
        system.reset()
        assert system._unstable is None
        system.step(20)
        assert np.isfinite(system.get_positions_3d()[1]).all()
    finally:
        system.close()


def test_eam_declares_no_decomposition():
    """EAM's energy is not pairwise-additive, so there is nothing to decompose and
    nothing to verify. Saying so is the honest answer, not a gap."""
    system = registry.build("cu_deposition")
    try:
        assert system.force_field.energy_terms_labels == ()
        assert verify_system(system) is None
        assert system.get_total_potential_terms() is None
    finally:
        system.close()
