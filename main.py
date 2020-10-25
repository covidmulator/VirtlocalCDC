import ray
from model import EpiNN
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

ray.init(num_gpus=2)
config = a3c.DEFAULT_CONFIG.copy()
VirtLocalCDC = EpiNN()
ModelCatalog.register_custom_model("CDC_model", VirtLocalCDC)

config["num_gpus"] = 2
config["num_workers"] = 10
config["eager"] = True
config["model"] = "CDC_model"


trainer = a3c.A3CTrainer(config=config,env=)

for i in range(1000):
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)



