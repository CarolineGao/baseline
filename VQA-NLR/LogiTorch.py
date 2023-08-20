import pytorch_lightning as pl
from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pl_models.ruletaker import PLRuleTaker
from torch.utils.data.dataloader import DataLoader

# Load the RuleTaker model from a checkpoint
model = PLRuleTaker.load_from_checkpoint("best_ruletaker.ckpt")

# Example natural language question
context = "Bob is smart. If someone is smart then he is kind"
question = "Bob is kind"

# Predict the logical operator
pred = model.predict(context, question)
print(RuleTakerDataset.ID_TO_LABEL[pred])
