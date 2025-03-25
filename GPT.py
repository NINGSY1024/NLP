from transformer import Encoder
from torch import nn,optim
from torch.nn.functional import cross_entropy,softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torch
import utils
import os
import pickle

class GPT(nn.Module):

    def __init__(self, model_dim, max_len, num_layer, num_head, n_vocab, lr, max_seg=3, drop_rate=0.2,padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len
        
        self.word_emb = nn.Embedding(n_vocab,model_dim)
        self.word_emb.weight.data.normal_(0,0.1)

        self.segment_emb = nn.Embedding(num_embeddings= max_seg, embedding_dim=model_dim)
        self.segment_emb.weight.data.normal_(0,0.1)
        self.position_emb = torch.empty(1,max_len,model_dim)
        nn.init.kaiming_normal_(self.position_emb,mode='fan_out', nonlinearity='relu')
        self.position_emb = nn.Parameter(self.position_emb)


        self.encoder = Encoder(n_head=num_head, emb_dim=model_dim, drop_rate=drop_rate, n_layer=num_layer)
        self.task_mlm = nn.Linear(in_features=model_dim, out_features=n_vocab)
        self.task_nsp = nn.Linear(in_features=model_dim*self.max_len, out_features=2)

        self.opt = optim.Adam(self.parameters(),lr)
    
    def forward(self,seqs, segs, training=False):
        embed = self.input_emb(seqs, segs)
        z = self.encoder(embed, training, mask = self.mask(seqs))   # [n, step, model_dim]
        mlm_logits = self.task_mlm(z)   # [n, step, n_vocab]
        nsp_logits = self.task_nsp(z.reshape(z.shape[0],-1))    # [n, n_cls]
        return mlm_logits, nsp_logits
    
    def step(self, seqs, segs, seqs_, nsp_labels):
        self.opt.zero_grad()
        mlm_logits, nsp_logits = self(seqs, segs, training=True)
        pred_loss = cross_entropy(mlm_logits.reshape(-1,self.n_vocab),seqs_.reshape(-1))
        nsp_loss = cross_entropy(nsp_logits,nsp_labels.reshape(-1))
        loss = pred_loss + 0.2 * nsp_loss
        loss.backward()
        self.opt.step()
        return loss.cpu().data.numpy(), mlm_logits
    
    def input_emb(self,seqs, segs):
        # device = next(self.parameters()).device
        # self.position_emb = self.position_emb.to(device)
        return self.word_emb(seqs) + self.segment_emb(segs) + self.position_emb
    
    def mask(self, seqs):
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        mask = torch.triu(torch.ones((seq_len,seq_len), dtype=torch.long), diagonal=1).to(device)  # [seq_len ,seq_len]
        pad = torch.eq(seqs,self.padding_idx)   # [n, seq_len]
        mask = torch.where(pad[:,None,None,:],1,mask[None,None,:,:]).to(device)   # [n, 1, seq_len, seq_len]
        return mask>0   # [n, 1, seq_len, seq_len]
    
    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.cpu().data.numpy() for l in self.encoder.encoder_layers]
        }
        return attentions

def evaluate(model, dataset, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in loader:
            seqs, segs, xlen, nsp_labels = batch
            seqs, segs, nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device), nsp_labels.to(device)
            mlm_logits, nsp_logits = model(seqs[:,:-1], segs[:,:-1], training=False)
            
            # Calculate MLM loss
            pred_loss = cross_entropy(mlm_logits.reshape(-1, model.n_vocab), seqs[:,1:].reshape(-1))
            
            # Calculate NSP accuracy
            nsp_preds = nsp_logits.argmax(dim=1)
            correct_predictions += (nsp_preds == nsp_labels.reshape(-1)).sum().item()
            total_predictions += len(nsp_labels)
            
            total_loss += pred_loss.item()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def visualize_attention(model, dataset, device, num_examples=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of examples
        seqs, segs, xlen, nsp_labels = dataset[:num_examples]
        seqs, segs, nsp_labels = torch.from_numpy(seqs), torch.from_numpy(segs), torch.from_numpy(nsp_labels)
        seqs, segs, nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device), nsp_labels.to(device)
        
        # Get model predictions and attention
        mlm_logits, nsp_logits = model(seqs[:,:-1], segs[:,:-1], training=False)
        attentions = model.attentions
        
        # Print examples with attention visualization
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print("Input:", " ".join([dataset.i2v[j] for j in seqs[i].cpu().numpy()[:xlen[i].sum()+1]]))
            print("NSP Prediction:", "Same" if nsp_logits[i].argmax().item() == 1 else "Different")
            print("NSP Ground Truth:", "Same" if nsp_labels[i].item() == 1 else "Different")
            print("-" * 80)

def train():
    MODEL_DIM = 512
    N_LAYER = 6
    LEARNING_RATE = 5e-5
    dataset = utils.MRPCData("./MRPC",2000)
    print("num word: ",dataset.num_word)
    model = GPT(
        model_dim=MODEL_DIM, max_len=dataset.max_len-1, num_layer=N_LAYER, num_head=8, n_vocab=dataset.num_word,
        lr=LEARNING_RATE, max_seg=dataset.num_seg, drop_rate=0.1, padding_idx=dataset.pad_id
    )
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    
    loader = DataLoader(dataset,batch_size=16,shuffle=True)
    best_loss = float('inf')

    for epoch in range(200):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            seqs, segs,xlen,nsp_labels = batch
            seqs, segs,nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),nsp_labels.to(device)
            loss,pred = model.step(seqs=seqs[:,:-1], segs= segs[:,:-1], seqs_=seqs[:,1:], nsp_labels=nsp_labels)
            total_loss += loss
            if batch_idx %100 == 0:
                pred = pred[0].cpu().data.numpy().argmax(axis = 1)
                print(
                    "Epoch: ",epoch,
                    "|batch: ", batch_idx,
                    "| loss: %.3f" % loss,
                    "\n| tgt: ", " ".join([dataset.i2v[i] for i in seqs[0, 1:].cpu().data.numpy()[:xlen[0].sum()+1]]),
                    "\n| prd: ", " ".join([dataset.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                )
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.3f}")
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, dataset, device)
        print(f"Validation Loss: {val_loss:.3f}, Accuracy: {val_accuracy:.3f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("./visual/models/gpt",exist_ok=True)
            torch.save(model.state_dict(),"./visual/models/gpt/model.pth")
    
    # Final evaluation and visualization
    print("\nFinal Model Evaluation:")
    final_loss, final_accuracy = evaluate(model, dataset, device)
    print(f"Final Loss: {final_loss:.3f}, Accuracy: {final_accuracy:.3f}")
    
    print("\nAttention Visualization:")
    visualize_attention(model, dataset, device)
    
    export_attention(model,device,dataset)

def export_attention(model,device,data,name="gpt"):
    model.load_state_dict(torch.load("./visual/models/gpt/model.pth",map_location=device))
    seqs, segs,xlen,nsp_labels = data[:32]
    seqs, segs,xlen,nsp_labels = torch.from_numpy(seqs),torch.from_numpy(segs),torch.from_numpy(xlen),torch.from_numpy(nsp_labels)
    seqs, segs,nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),nsp_labels.to(device)
    model(seqs[:,:-1],segs[:,:-1],False)
    seqs = seqs.cpu().data.numpy()
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
if __name__ == "__main__":
    train()
            



