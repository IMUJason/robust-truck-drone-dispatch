"""DAGCD — Dynamic Air-Ground Collaborative Delivery research package."""

from .instance import (Instance, Solution, Node, Order, GroundVehicle, Drone, NFZ,
                       WeatherEvent, WEATHER_STATES, GroundRoute, GroundStop,
                       DroneSortie, DroneStop,
                       load_instance, load_solution, instance_to_dict)
from .checker import check_solution, CheckReport, CostParams
from .generator import (InstanceParams, Scenario, sample_instance, sample_scenario,
                        snapshot_at, scenario_to_dict, scenario_from_dict,
                        save_scenario, load_scenario, perturb_params)
from .simulator import (run_simulation, GreedyController, SimReport, FleetState,
                        FleetView, EpochRecord)
from .oracle import (InsertionOracle, MipOracle, CplexOracle, enumerate_sorties)
from .audit import (AuditReport, adversarial_regret_audit, _quantile_ub_index,
                    audit_to_dict, audit_from_dict, save_audit, load_audit)
from .evolve import (EvolutionError, ProgramError, LLMClient,
                     OpenAICompatibleClient, MockLLM, compile_controller,
                     probe_controller, FitnessConfig, EvalResult,
                     evaluate_program, clear_eval_cache, EvolutionConfig,
                     EvolutionResult, run_evolution, SEED_TEMPLATE,
                     DEFAULT_MUTATION_OPERATORS)
