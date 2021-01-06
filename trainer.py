# +
import logging

logger = logging.getLogger(__name__)


# -

class Trainer:
    """abstract class for trainer"""


from models import EMAModel
from metric import Collection, Loss
from dataclasses import dataclass


# ### Mean Teacher

@dataclass
class Metrics(Collection):
    loss: Loss.from_output("loss")
    sup_loss: Loss.from_output("sup_loss")
    sup_cst_loss: Loss.from_output("sup_cst_loss")


# +

# sup_loss_fn = nn.CrossEntropyLoss(reduction="none")  # "none"
# consistency_fn = softmax_mse_loss

class MeanTeacherTrainer(Trainer):
            
    def train(self, n_epochs, model, train_dl, loss_fn, consistency_loss_fn, optimizer, lr, **kwargs):
        assert "cst_factor" in kwargs
        cst_factor = kwargs["cst_factor"]
        
        optimizer = self.optimizer(model.parameters(), lr=lr)
        
        metrics = None
        for epoch in range(n_epochs):
            model.train() ; ema_model.train()

            for x, x_lens, xa, xa_lens, y in train_dl:
                x, xa, y = x.cuda(), xa.cuda(), y.cuda()

                # SUPERVISED LOSS
                logits = model(x, x_lens)
                sup_loss = loss_fn(logits, y).mean()

                # CONSISTENCY LOSS
                logits_aug = ema_model(xa, xa_lens).detach() # no backprop for teacher
                
                sup_cst_loss = cst_factor * consistency_loss_fn(logits, logits_aug, True).mean()
                loss = sup_loss +  sup_cst_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)
                optimizer.step() ; optimizer.zero_grad()
                ema_model.update_parameters(ema_decay, global_step)

                update = Metrics.single_from_model_output(
                    loss=loss, 
                    sup_loss=sup_loss, 
                    sup_cst_loss=sup_cst_loss
                )
                
                with torch.no_grad():
                    metrics = update if metrics is None else metrics.merge(update)
                logger.info(f"\t[{epoch}][{i}] Loss: {metrics.compute()}")


# -

# ### UDA

class UDATrainer(Trainer):
    
    def get_tsa_threshold(self, schedule, global_step, num_train_steps, start, end):
        training_progress = torch.tensor(float(global_step) / float(num_train_steps))
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = torch.exp( (training_progress - 1) * scale )
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - torch.exp(-training_progress * scale)
        output = threshold * (end - start) + start
        return output
            
    
    def train(self, n_epochs, model, train_dl, unlabeled_dl, loss_fn, consistency_loss_fn, unsup_loss_fn, optimizer, lr, **kwargs):
        optimizer = self.optimizer(model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            _sup_loss = 0. ; _unsup_loss = 0. 
            _sup_cst_loss = 0. ; _unsup_cst_loss = 0.
            _sup_acc = 0. ; _unsup_acc = 0.
            _sup_agreement = 0. ; _unsup_agreement = 0. ;
            _unsup_ce_loss = 0. ; _unsup_ce_loss_au = 0.
            _n_sup = 0 ; _n_unsup = 0
            model.train()

            for i, sample in enumerate(train_dl):
                x, x_lens, xa, xa_lens, y = sample
                x=x.cuda() ; y=y.cuda() ; xa=xa.cuda()

                try:
                    x_un, x_un_lens, xa_un, xa_un_lens, y_un = next(unlabeled_it)
                except StopIteration:
                    unlabeled_it = iter(unlabeled_dl)
                    x_un, x_un_lens, xa_un, xa_un_lens, y_un = next(unlabeled_it)
                x_un=x_un.cuda(); xa_un=xa_un.cuda(); y_un=y_un.cuda()

                logits = model(x, x_lens)
                logits_au = model(xa, xa_lens)
                sup_cst_loss = sup_cst * consistency_loss_fn(logits, logits_au, True)

                sup_loss = loss_fn(logits, y)
                # === TSA ===
                global_step += 1
                tsa_threshold = self.get_tsa_threshold(tsa_schedule, global_step, n_warmup, start=1/n_classes+0.02, end=.75).cuda()
                larger_than_threshold = torch.exp(-sup_loss) > tsa_threshold
                loss_mask = 1. - larger_than_threshold.float()
                loss = 0.
                # broadcasting with loss mask: should we do loss mask
                n_sup = loss_mask.sum()
                if n_sup > 0:
                    sup_loss = (sup_loss * loss_mask).sum(-1) / n_sup
                    sup_cst_loss = (sup_cst_loss * loss_mask).sum(-1) / n_sup
                    loss += sup_loss + sup_cst_loss

                # === UNSUPERVISED ===
                logits_un = model(x_un, x_un_lens)
                logits_un_au = model(xa_un, xa_un_lens)

                prob_un = logits_un.softmax(-1)
                unsup_cst_loss = unsup_cst*consistency_loss_fn(logits_un, logits_un_au, True)
                # loss += unsup_cst_loss

                # Confidence based masking for unlabeled
                # only release examples with conf > min threshold
                if uda_conf_threshold > 0.:
                    unsup_loss_mask = (prob_un.max(-1)[0] > uda_conf_threshold).float()
                else:
                    unsup_loss_mask = torch.ones(logits_un.size(0)).cuda()

                # Sharpening for unlabeled aug
                # prob_aug = (logits_aug_un.softmax(-1) / sharpen_T).log_softmax(-1)
                # prob_aug = sharpen(logits_aug_un.softmax(-1), sharpen_T).exp().log_softmax(-1)
                # prob_aug = logits_aug_un / uda_softmax_temp

                # Pseudo-labels
                # original data is relatively better predictor
                prob_un = sharpen(prob_un, sharpen_T)
                unsup_loss = unsup_loss_fn(logits_un_au.log_softmax(-1), prob_un).sum(-1)

                # avoid / by 0
                n_unsup = unsup_loss_mask.sum()
                if n_unsup > 0.:
                    unsup_loss = (unsup_loss * unsup_loss_mask).sum(-1) / n_unsup
                    unsup_cst_loss = (unsup_cst_loss * unsup_loss_mask).sum(-1)/n_unsup
                    unsup_loss *= uda_coeff
                    loss += unsup_loss + unsup_cst_loss

                # unsup_cst_loss
                if loss > 0.:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)
                    optimizer.step() ; optimizer.zero_grad()
                with torch.no_grad():
                    _sup_loss += sup_loss.item() if n_sup > 0 else 0.
                    _unsup_loss += unsup_loss.item() if n_unsup > 0 else 0. # loss_fn(logits_un, y_un).mean().item()
                    _sup_cst_loss += sup_cst_loss.item() if n_sup > 0 else 0.
                    _unsup_cst_loss += unsup_cst_loss.item() if n_unsup > 0 else 0.
                    _sup_acc += (logits.argmax(-1) == y).float().mean().item()
                    _unsup_acc += (logits_un.argmax(-1) == y_un).float().mean().item()
                    _sup_agreement += (logits.argmax(-1) == logits_au.argmax(-1)).float().mean().item()
                    _unsup_agreement += (logits_un.argmax(-1) == logits_un_au.argmax(-1)).float().mean().item()
                    _n_sup += n_sup.item()
                    _n_unsup += n_unsup.item()
                    _unsup_ce_loss += loss_fn(logits_un, y_un).mean().item()
                    _unsup_ce_loss_au += loss_fn(logits_un_au, y_un).mean().item()
            _sup_loss /= (i+1) ; _unsup_loss /= (i+1) ; _sup_cst_loss /= (i+1)
            _unsup_cst_loss /= (i+1) ; _sup_acc /= (i+1) ; _unsup_acc /= (i+1)
            _sup_agreement /= (i+1) ; _unsup_agreement /= (i+1)
            _n_sup /= (i+1) ; _n_unsup /= (i+1)
            _unsup_ce_loss /= (i+1) ; _unsup_ce_loss_au /= (i+1)
            msg.info(f"\t[{i}] Loss Sup: {_sup_loss:.4f} - Unsup: {_unsup_loss:.4f}")
            msg.info(f"\tCST Sup: {_sup_cst_loss:.4f} - Unsup: {_unsup_cst_loss:.4f}")
            msg.info(f"\tAcc Sup: {_sup_acc:.4f} - Unsup: {_unsup_acc:.4f}")
            msg.info(f"\tAgreement Sup: {_sup_agreement:.4f} - Unsup: {_unsup_agreement:.4f}")
            msg.info(f"\tTSA Num Exs Sup: {_n_sup:.2f} - Unsup: {_n_unsup:.2f} - threshold: {tsa_threshold.item():.2f}")
            msg.info(f"\tUnsup CE: {_unsup_ce_loss:.4f} - CE au: {_unsup_ce_loss_au:.4f}")

            best_val, patience, _val_loss, _val_acc = val_step(model, model_name, val_dl, best_val, patience)
            msg.info(f"\tElapsed: {time.time() - start:.4f}")

            train_sup_loss += [_sup_loss] ; train_unsup_loss += [_unsup_loss]
            train_sup_cst_loss += [_sup_cst_loss] ; train_unsup_cst_loss += [_unsup_cst_loss]
            train_sup_acc += [_sup_acc] ; train_unsup_acc += [_unsup_acc]
            train_sup_agreement += [_sup_agreement] ; train_unsup_agreement += [_unsup_agreement]
            val_loss += [_val_loss] ; val_acc += [_val_acc]
            train_unsup_ce_loss += [_unsup_ce_loss] ; train_unsup_ce_loss_au += [_unsup_ce_loss_au]
            train_n_sup += [_n_sup] ; train_n_unsup += [_n_unsup]
            if patience > max_patience: raise StopIteration()


class MixMatchTrainer(Trainer):
    
    def train(self, n_epochs, model, train_dl, unlabeled_dl, loss_fn, consistency_loss_fn, unsup_loss_fn, optimizer, lr, **kwargs):

        best_val = 1e+4 ; max_patience = 70 ; patience = 0
        # METRIC
        train_loss = [] ; train_unsup_acc = []
        train_n_unsup = []
        val_loss = [] ; val_acc = []

        unlabeled_it = iter(unlabeled_dl)
        n_warmup = len(train_dl) * 100
        global_step = 0

        for epoch in range(n_epochs):
            start = time.time()
            msg.divider()
            msg.info(f"\t=== EPOCH {epoch} ===")
            _loss = 0. ; _unsup_acc = 0. ; _n_unsup=0.
            model.train()

            for i, sample in enumerate(train_dl):
                global_step += 1
                x, x_lens, xa, xa_lens, y_ = sample
                x=x.cuda() ; y_=y_.cuda() ; xa=xa.cuda()

                try:
                    x_un, x_un_lens, xa_un, xa_un_lens, y_un_ = next(unlabeled_it)
                except StopIteration:
                    unlabeled_it = iter(unlabeled_dl)
                    x_un, x_un_lens, xa_un, xa_un_lens, y_un_ = next(unlabeled_it)
                x_un=x_un.cuda(); xa_un=xa_un.cuda(); y_un_=y_un_.cuda()
                y = F.one_hot(y_, n_classes).float()
                y_un = F.one_hot(y_un_, n_classes).float()

                # Pseudo-labels
                # original data is relatively better predictor
                model.eval() # turn off drop out
                prob_un = model(x_un, x_un_lens).softmax(-1).detach()

                # Confidence based masking for unlabeled
                # only release examples with conf > min threshold
                unsup_loss_mask = (prob_un.max(-1)[0] > unlabeled_conf)

                prob_un = sharpen(prob_un, sharpen_T).detach()
                model.train()

                n_unsup = unsup_loss_mask.sum()
                if n_unsup > 0.:
                    # threshold the growth
                    if epoch > 30:
                        growth_factor = 1.1
                    else:
                        growth_factor = 0.
                    max_unsup = int(min(max(train_n_unsup[-1], 2) * growth_factor, n_unsup)) + 1
                    max_unsup = 1 if max_unsup < 2 else max_unsup
                    n_unsup = max_unsup

                    x_un = x_un[unsup_loss_mask][:n_unsup]
                    prob_un = prob_un[unsup_loss_mask][:n_unsup]
                    x_un_lens = [x_un_lens[j] 
                                 for j, k in enumerate(unsup_loss_mask) 
                                 if k == True][:n_unsup]
                    # concat data
                    y_all = torch.cat([y, prob_un], dim=0)

                    # Padding
                    max_seq_len = max(x.size(1), x_un.size(1))
                    if max_seq_len > x.size(1):
                        pad = x.new_zeros(x.size(0), max_seq_len - x.size(1))
                        x = torch.cat([x, pad], dim=1)
                    else:
                        pad = x_un.new_zeros(x_un.size(0), max_seq_len - x_un.size(1))
                        x_un = torch.cat([x_un, pad], dim=1)
                    assert x.size(1) == x_un.size(1)
                    x_all = torch.cat([x, x_un], dim=0)
                    x_lens = x_lens + x_un_lens
                else:
                    x_all = x
                    y_all = y

                l = np.random.beta(alpha, alpha)
                l = max(l, 1-l)

                idx = torch.randperm(x_all.size(0))
                x1, x2 = x_all, x_all[idx]
                y1, y2 = y_all, y_all[idx]
                x1_lens, x2_lens = x_lens, [x_lens[j] for j in idx]
                # mix target
                y_mix = l * y1 + (1-l) * y2

                logits_mix = model(x1, x1_lens, x2, x2_lens, l)
                loss = loss_fn(logits_mix.log_softmax(-1), y_mix).sum(-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)
                optimizer.step() ; optimizer.zero_grad()
                with torch.no_grad():
                    _loss += loss.item()
                    _unsup_acc = (prob_un.argmax(-1) == y_un_[unsup_loss_mask][:prob_un.size(0)]).float().mean().item() if n_unsup > 0 else 0.
                    _n_unsup += n_unsup

            _loss /= (i+1) ; _n_unsup /= (i+1) ; _unsup_acc /= (i+1)
            msg.info(f"\t[{i}] Loss Sup: {_loss:.4f} - Acc unsup: {_unsup_acc:.4f}")
            msg.info(f"\tN unsup: {_n_unsup:.4f}")

            best_val, patience, _val_loss, _val_acc = val_step(model, model_name, val_dl, best_val, patience)
            msg.info(f"\tElapsed: {time.time() - start:.4f}")

            train_loss += [_loss] ; train_unsup_acc += [_unsup_acc]
            train_n_unsup += [_n_unsup]
            val_loss += [_val_loss] ; val_acc += [_val_acc]
            if patience > max_patience: raise StopIteration()


