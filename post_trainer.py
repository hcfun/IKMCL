import torch
from torch import optim
import numpy as np
from utils import get_posttrain_train_valid_dataset
from torch.utils.data import DataLoader
from datasets import KGETrainDataset, KGEEvalDataset
from trainer import Trainer


class PostTrainer(Trainer):
    def __init__(self, args):
        super(PostTrainer, self).__init__(args)
        self.args = args
        self.load_metatrain()

        # dataloader
        train_dataset, valid_dataset = get_posttrain_train_valid_dataset(args)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.posttrain_bs,
                                      collate_fn=KGETrainDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.indtest_eval_bs,
                                      collate_fn=KGEEvalDataset.collate_fn)

        self.optimizer = optim.Adam(list(self.ent_init.parameters()) + list(self.compgcn.parameters())
                                    + list(self.kge_model.parameters()), lr=self.args.posttrain_lr)

    def load_metatrain(self):
        state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.compgcn.load_state_dict(state['compgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        ent_emb=self.compgcn(sup_g_bidir)
        return ent_emb



    def train(self):
        self.logger.info('start fine-tuning')

        # print epoch test rst
        self.evaluate_indtest_test_triples(num_cand=50)

        for i in range(1, self.args.posttrain_num_epoch + 1):
            losses = []
            for batch in self.train_dataloader:                                                  #[d.to(self.args.gpu) for d in data[1:]]
                pos_triple, neg_tail_ent, neg_head_ent,que_pos_tail_ent,que_pos_head_ent,weight = [b.to(self.args.gpu) for b in batch]

                ent_emb=self.get_ent_emb(self.indtest_train_g)

                loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb,weight)
                cl_loss1 = self.get_cl_loss(pos_triple, que_pos_tail_ent, que_pos_head_ent, ent_emb)
                loss=loss+cl_loss1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            self.logger.info('epoch: {} | loss: {:.4f}'.format(i, np.mean(losses)))

            if i % self.args.posttrain_check_per_epoch == 0:
                self.evaluate_indtest_test_triples(num_cand=50)

    def evaluate_indtest_valid_triples(self, num_cand='all'):
        ent_emb = self.get_ent_emb(self.indtest_train_g)

        results = self.evaluate(ent_emb, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
