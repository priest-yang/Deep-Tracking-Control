
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from .base.legged_robot import LeggedRobot
from .lite3.lite3_rough_config import Lite3RoughCfg, Lite3RoughCfgPPO
from .lite3.lite3_fast_config import Lite3FastCfg, Lite3FastCfgPPO
from .x30.x30_rough_config import X30RoughCfg, X30RoughCfgPPO

from legged_gym.utils.task_registry import task_registry

from .lite3.lite3_dtc_config import Lite3DTCCfgPPO, Lite3DTCCfg
from .base.legged_robot_dtc import LeggedRobotDTC

task_registry.register( "lite3_rough", LeggedRobot, Lite3RoughCfg(), Lite3RoughCfgPPO() )
task_registry.register( "x30_rough", LeggedRobot, X30RoughCfg(), X30RoughCfgPPO() )

task_registry.register( "lite3_fast", LeggedRobot, Lite3FastCfg(), Lite3FastCfgPPO() )

task_registry.register( "lite3_dtc", LeggedRobotDTC, Lite3DTCCfg(), Lite3DTCCfgPPO() )