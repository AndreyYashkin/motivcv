from typing import override
from pytorch_lightning.callbacks import Callback

try:
    from clearml import Task

    _clearml_av = True
except ImportError:
    _clearml_av = False


class ClearmlTask(Callback):
    def __init__(self, project_name, task_name, tags=[]):
        assert _clearml_av, "Clearml cannot be imported!"
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags
        self.task = None

    @override
    def on_fit_start(self, trainer, pl_module):
        self.task = Task.init(
            project_name=self.project_name, task_name=self.task_name, tags=self.tags
        )

    @override
    def on_fit_end(self, trainer, pl_module):
        self.task.close()
        self.task = None
