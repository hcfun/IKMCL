from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
from cl_loss import CL, distance, MarginLoss, NegativeSamplingLoss, Contrast, ContrastiveLoss, C_loss
from compgcn import ExtGNN
from utils import Log
from torch.utils.data import DataLoader
from ent_init_model import EntInit
from rgcn_model import RGCN
from kge_model import KGEModel
import torch
import torch.nn.functional as F
from collections import defaultdict as ddict
from utils import get_indtest_test_dataset_and_train_g
from datasets import KGEEvalDataset




class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cl_net = CL(args.temperature).cuda()
        self.conloss=ContrastiveLoss()
        self.c_loss = C_loss(args.temperature).cuda()
        self.ContrastiveLoss = Contrast(args.ent_dim, args.temperature, 0.5).cuda()
        self.margin_loss = MarginLoss(margin=1.0).cuda()
        self.nrelation = args.num_rel
        self.nents=args.num_ent
        self.epsilon = 2.0
        self.neg_sampling_loss = NegativeSamplingLoss().cuda()
        self.gamma = torch.Tensor([args.gamma])
        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.emb_dim])
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, args.rel_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.ent_embedding = nn.Parameter(torch.zeros(self.nents, args.emb_dim))
        nn.init.uniform_(
            tensor=self.ent_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )


        # writer and logger
        self.name = args.name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        indtest_test_dataset, indtest_train_g = get_indtest_test_dataset_and_train_g(args)
        self.indtest_train_g = indtest_train_g.to(args.gpu)
        self.indtest_test_dataloader = DataLoader(indtest_test_dataset, batch_size=args.indtest_eval_bs,
                                                  shuffle=False, collate_fn=KGEEvalDataset.collate_fn)

        # models
        self.ent_init = EntInit(args).to(args.gpu)
        self.rgcn = RGCN(args).to(args.gpu)
        self.compgcn = ExtGNN(args).to(args.gpu)
        self.kge_model = KGEModel(args).to(args.gpu)


    def save_checkpoint(self, step):
        state = {'ent_init': self.ent_init.state_dict(),
                 'compgcn': self.compgcn.state_dict(),
                 'kge_model': self.kge_model.state_dict()}
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.compgcn.load_state_dict(state['compgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def get_loss(self, tri, neg_tail_ent, neg_head_ent,ent_emb,weight):
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')

        neg_score = torch.cat([neg_head_score, neg_tail_score])
        # neg_head_score = (F.softmax(neg_head_score * self.args.adv_temp, dim=1).detach()
        #              * F.logsigmoid(-neg_head_score)).sum(dim=1)
        # neg_tail_score = (F.softmax(neg_tail_score * self.args.adv_temp, dim=1).detach()
        #                   * F.logsigmoid(-neg_tail_score)).sum(dim=1)
        weight = weight.cuda()
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                          * F.logsigmoid(-neg_score)).sum(dim=1)
        weight2=torch.cat([weight,weight])

        # pos_tail_score = self.kge_model((tri, pos_tail_ent), ent_emb, mode='tail-batch')
        # pos_head_score = self.kge_model((tri, pos_head_ent), ent_emb, mode='head-batch')
        # pos2_score = torch.cat([pos_tail_score, pos_head_score])
        # pos2_score = (F.softmax(pos2_score * self.args.adv_temp, dim=1).detach()
        #              * F.logsigmoid(-pos2_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)


        positive_sample_loss = - (weight * pos_score).sum() / weight.sum()
        #negative_sample_loss1 = - (weight * neg_head_score).sum() / weight.sum()
        negative_sample_loss2 = - (weight2 * neg_score).sum() / weight2.sum()
        neg_loss=negative_sample_loss2


        # positive_sample_loss = - pos_score.mean()
        # negative_sample_loss = - neg_score.mean()
        #pos_sample_loss = - pos2_score.mean()
        loss = (positive_sample_loss + neg_loss)/2



        return loss

    def get_cl_loss(self, tri, pos_tail_ent, pos_head_ent, ent_emb):
        labels1 = tri[:, 0]
        labels2 = tri[:, 1]
        labels3 = tri[:, 2]
        head_emb = torch.index_select(
            ent_emb,
            dim=0,
            index=tri[:,0]
        ).unsqueeze(1)

        tail_emb = torch.index_select(
            ent_emb,
            dim=0,
            index=tri[:, 2]
        ).unsqueeze(1)

        # head_emb2 = torch.index_select(
        #     self.ent_embedding.cuda(),
        #     dim=0,
        #     index=tri[:, 0]
        # ).unsqueeze(1)
        # tail_emb2 = torch.index_select(
        #     self.ent_embedding.cuda(),
        #     dim=0,
        #     index=tri[:, 2]
        # ).unsqueeze(1)


        # head_emb2 = torch.index_select(
        #     que_emb,
        #     dim=0,
        #     index=tri[:, 0]
        # ).unsqueeze(1)
        #
        # tail_emb2 = torch.index_select(
        #     que_emb,
        #     dim=0,
        #     index=tri[:, 2]
        # ).unsqueeze(1)


        pos_tail_emb = torch.index_select(
            ent_emb,
            dim=0,
            index=pos_tail_ent.squeeze(dim=1)
        ).unsqueeze(1)
        pos_head_emb = torch.index_select(
            ent_emb,
            dim=0,
            index=pos_head_ent.squeeze(dim=1)
        ).unsqueeze(1)

        cl_loss2 = self.cl_net(tail_emb, pos_tail_emb,labels3)
        #cl_loss1 = self.cl_net(head_emb, pos_head_emb, labels1)


        #cl_loss=self.ContrastiveLoss(tail_emb.squeeze(1),pos_tail_emb.squeeze(1),labels3)
        #cl_loss2 = self.ContrastiveLoss(head_emb.squeeze(1), pos_head_emb.squeeze(1), labels1)
        #cl_loss = self.ContrastiveLoss(head_emb.squeeze(1), head_emb2.squeeze(1), labels1)
        # pos_distance = distance(head_emb + relation, tail_emb)
        # neg_distance = distance(head_emb + relation, neg_tail_emb)
        # neg_distance2 = distance(neg_head_emb + relation, tail_emb)
        #
        # # 计算损失
        # cl_loss2 = self.margin_loss(pos_distance, neg_distance)
        # cl_loss1 = self.neg_sampling_loss(pos_distance, neg_distance)
        # cl_loss3 = self.neg_sampling_loss(pos_distance, neg_distance2)


        return cl_loss2

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        #在这里加上子图对比
       # sup_g_bidir{dstdata{feat:[5232,32]},ndata{'feat':[5232,32]},srcdata{'feat':[5232,32]}}
        ent_emb=self.compgcn(sup_g_bidir)
        return ent_emb




    def evaluate(self, ent_emb, eval_dataloader, num_cand='all'):
        results = ddict(float)
        count = 0

        eval_dataloader.dataset.num_cand = num_cand

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        else:
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        return results

    def evaluate_indtest_test_triples(self, num_cand='all'):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb = self.get_ent_emb(self.indtest_train_g)

        results = self.evaluate(ent_emb,self.indtest_test_dataloader, num_cand=num_cand)

        self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results












