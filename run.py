import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import time
from torch.utils.data import Dataset
from model import *
from params import *


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

class BlockDataset(Dataset):
    def __init__(self):
        self.blocks = [(torch.randint(1, vocab_size, (MAX_LEN,)), torch.randint(1, vocab_size, (MAX_LEN,))) for i in range(15)]
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        x, y = self.blocks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def _mp_fn(index):
  device = xm.xla_device()
  
  train_ds = BlockDataset()
  train_ds = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

  mp_device_loader = pl.MpDeviceLoader(train_loader, device)
  
  model = Transformer(
      n_layers=N_LAYERS, 
      n_heads=N_HEADS, 
      n_embd=N_EMBD, 
      vocab_size=vocab_size, 
      num_dense_layers=NUM_DENSE_LAYER, 
      num_expert=N_EXPERT, 
      score_fn=SCORE_FN, 
      top_k=N_TOP_K, 
      inter_size=N_INTER_SIZE,
      max_len=MAX_LEN,
  ).to(device)
  
  print(f"model has {sum(p.numel() for p in model.parameters()) / 1e6} M")
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
  
  
  model.train()
  N = 5
  begin = time.time()
  for i in range(N):
      men = []
      for x, y in train_ds:
          b = time.time()
          x, y = x.to(device), y.to(device)
          with autocast(device_type='xla', dtype=torch.bfloat16):
              optimizer.zero_grad()
              out, aux_loss = model(x)
              loss = F.cross_entropy(
                  out.view(-1, out.size(-1)),
                  y.view(-1)
              )
              loss += aux_loss
              loss.backward()
              optimizer.step()
              xm.mark_step()
          e = time.time()
          men.append(e - b)
      print(f"epoch: {i} - time_per_batch: {sum(men)/len(men)}")
  
  end = time.time()
  
  print(f'total_running_time : {end - begin}')


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
  
    
