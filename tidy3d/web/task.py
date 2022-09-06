""" Defnes information about a task """
from datetime import datetime
from enum import Enum
from abc import ABC
from typing import Optional

import pydantic


class TaskStatus(Enum):
    """the statuses that the task can be in"""

    INIT = "initialized"
    QUEUE = "queued"
    PRE = "preprocessing"
    RUN = "running"
    POST = "postprocessing"
    SUCCESS = "success"
    ERROR = "error"


class TaskBase(pydantic.BaseModel, ABC):
    """base config for all task objects"""

    class Config:
        """configure class"""

        arbitrary_types_allowed = True


# type of the task_id
TaskId = str

# type of task_name
TaskName = str


class TaskInfo(TaskBase):
    """general information about task"""

    taskId: str
    taskName: Optional[str] = None
    nodeSize: Optional[int] = None
    completedAt: Optional[datetime] = None
    status: Optional[str] = None
    realCost: Optional[float] = None
    timeSteps: Optional[int] = None
    solverVersion: Optional[str] = None
    createAt: Optional[datetime] = None
    estCostMin: Optional[float] = None
    estCostMax: Optional[float] = None
    realFlexUnit: Optional[float] = None
    estFlexUnit: Optional[float] = None
    s3Storage: Optional[float] = None
    startSolverTime: Optional[datetime] = None
    finishSolverTime: Optional[datetime] = None
    totalSolverTime: Optional[int] = None
    callbackUrl: Optional[str] = None
    taskType: Optional[str] = None


class RunInfo(TaskBase):
    """information about the run"""

    perc_done: pydantic.confloat(ge=0.0, le=100.0)
    field_decay: pydantic.confloat(ge=0.0, le=1.0)

    def display(self):
        """print some info"""
        print(f" - {self.perc_done:.2f} (%) done")
        print(f" - {self.field_decay:.2e} field decay from max")


class Folder(pydantic.BaseModel):
    """
    Folder information of a task
    """

    projectName: Optional[str] = None
    projectId: Optional[str] = None

    class Config:
        """configure class"""

        arbitrary_types_allowed = True
