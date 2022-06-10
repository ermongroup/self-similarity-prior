### Config structure

First level (`config/`): dataset-specific, task-agnostic information e.g. resolution, number of channels.

Second level (`config/task`): task-specific information e.g. eval strategies, shared optimization hyperparameters.

Third level (`config/task/model/`): model hyperparameters.
