"""LargeMap compatibility namespace; safe to import without Isaac."""
import os
CONFIG_PATH=os.path.join(os.path.dirname(__file__), '..', 'config')
def init_simulation_app(cfg):
    from isaacsim import SimulationApp
    import isaacsim
    experience=os.path.join(os.path.dirname(isaacsim.__file__),'apps','isaacsim.exp.base.python.kit')
    extra=os.environ.get('NAVVEL_KIT_EXTRA_ARGS','').split()
    return SimulationApp({'headless':bool(cfg['headless']),'anti_aliasing':1,'extra_args':extra},experience=experience)
