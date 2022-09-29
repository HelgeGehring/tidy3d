from tests.utils import SIM_CONVERT
from tidy3d import web
from tidy3d.web.config import DEFAULT_CONFIG

PATH_SIM_DATA = "tests/tmp/sim_data.hdf5"
for i in range(1):
    sim_data = web.run(simulation=SIM_CONVERT, task_name="test_for_ssl_disable", path=PATH_SIM_DATA)
