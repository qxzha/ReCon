import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model.SGRAF import SGRAF
from sklearn.mixture import GaussianMixture

import numpy as np
from scipy.spatial.distance import cdist
from utils import AverageMeter


class ReCon(nn.Module):
    def __init__(self, opt):
        super(ReCon, self).__init__()
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.similarity_model = SGRAF(opt)
        self.params = list(self.similarity_model.params)
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        self.step = 0
        self.KLDivLoss = nn.KLDivLoss(reduction='none')

    def state_dict(self):
        return self.similarity_model.state_dict()

    def load_state_dict(self, state_dict):
        self.similarity_model.load_state_dict(state_dict)

    def train_start(self):
        """switch to train mode"""
        self.similarity_model.train_start()

    def val_start(self):
        """switch to valuate mode"""
        self.similarity_model.val_start()

    def forward_sim_pair(self, img, img2, eps=1e-8):
        img_norm = torch.norm(img, p=2, dim=1).unsqueeze(0)
        img2_norm = torch.norm(img2, p=2, dim=1).unsqueeze(1)

        img = img.transpose(0, 1)
        sim_t = torch.mm(img2, img)
        sim_norm = torch.mm(img2_norm, img_norm)
        cos_sim = sim_t / sim_norm.clamp(min=eps)
        return cos_sim

    def info_nce_loss(self, similarity_matrix, temp=None):
        if temp is None:
            temp = self.opt.temp
        labels = torch.eye(len(similarity_matrix)).float().cuda()

        # select and combine multiple positives
        pos = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        # preds = pos
        pos = torch.exp(pos / temp).squeeze(1)

        # select only the negatives the negatives
        neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        neg = torch.exp(neg / temp)

        neg_sum = neg.sum(1)

        loss = -(torch.log(pos / (pos + neg_sum)))
        preds = pos / (pos + neg_sum)
        return loss, preds

    def forward_relation(self, relation_imgs, relation_caps, cross_sims, length=None, mode="v", eps=1e-6):
        index = cross_sims.argmax(-1).detach().cpu().numpy().tolist()
        rows = [[i] * length for i in index]
        cols = [index] * length

        if mode == "v":
            pseudo_relation_imgs = nn.Softmax(dim=-1)(relation_caps[rows, cols])
            relation_imgs = nn.Softmax(dim=-1)(relation_imgs)
            iais_loss = self.KLDivLoss(torch.log(relation_imgs + eps), pseudo_relation_imgs) + self.KLDivLoss(torch.log(pseudo_relation_imgs + eps), relation_imgs)
        elif mode == "t":
            pseudo_relation_caps = nn.Softmax(dim=-1)(relation_caps[rows, cols])
            relation_caps = nn.Softmax(dim=-1)(relation_caps)
            iais_loss = self.KLDivLoss(torch.log(relation_caps + eps), pseudo_relation_caps) + self.KLDivLoss(torch.log(pseudo_relation_caps + eps), relation_caps)
        else:
            print("error")

        return iais_loss
    
    def forward_sim_pair(self, img, img2, eps=1e-8):
        img_norm = torch.norm(img, p=2, dim=1).unsqueeze(0)
        img2_norm = torch.norm(img2, p=2, dim=1).unsqueeze(1)

        img = img.transpose(0, 1)
        sim_t = torch.mm(img2, img)
        sim_norm = torch.mm(img2_norm, img_norm)
        cos_sim = sim_t / sim_norm.clamp(min=eps)
        return cos_sim

    def warmup_batch(self, images, captions, lengths, ids, corrs, tau=0.01):
        self.step += 1
        batch_length = images.size(0)
        img_embs, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        targets_batch = self.similarity_model.targets[ids]
        sims = self.similarity_model.forward_sim(img_embs, cap_embs, cap_lens, 'sim')

        if self.opt.contrastive_loss == 'Triplet':
            loss_cl = self.similarity_model.forward_loss(sims)
        elif self.opt.contrastive_loss == 'InfoNCE':
            loss_cl, preds = self.info_nce_loss(sims)
        loss_rank = loss_cl.mean()

        img_glo, cap_glo = self.similarity_model.sim_enc.forward_emb_glo(img_embs, cap_embs, cap_lens)

        sim_img = self.forward_sim_pair(img_glo, img_glo)
        sim_text = self.forward_sim_pair(cap_glo, cap_glo)

        sim_img = nn.Softmax(dim=-1)(sim_img) / tau
        sim_text = nn.Softmax(dim=-1)(sim_text) / tau

        # loss_relation = self.KLDivLoss(sim_img.log(), sim_text) + self.KLDivLoss(sim_text.log(), sim_img)

        # loss = loss_rank + loss_relation

        loss = loss_rank

        self.optimizer.zero_grad()
        # loss = loss.mean()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss', loss.item(), batch_length)

    def train_batch(self, images, captions, attn_mask, lengths, ids, corrs, is_pseudo=1, is_noisy=1, is_lambda=1):
        self.step += 1
        batch_length = images.size(0)

        targets_batch = self.similarity_model.targets[ids]

        img_embs, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        sims = self.similarity_model.forward_sim(img_embs, cap_embs, cap_lens, 'sim')
        loss_rank, preds = self.info_nce_loss(sims)
        loss_rank_t, preds_t = self.info_nce_loss(sims.t())
        loss_rank = (loss_rank + loss_rank_t) * 0.5 * targets_batch
        preds = (preds + preds_t) * 0.5

        # sims_img, sims_cap = self.similarity_model.sim_enc.forward_topology(img_embs, cap_embs, cap_lens)
        # sims_img = torch.sigmoid(sims_img) * targets_batch
        # sims_cap = torch.sigmoid(sims_cap) * targets_batch
        # sims_topo = sims_img @ sims_cap.t()
        # loss_topo, _ = self.info_nce_loss(sims_topo, temp=1.0)
        # loss_topo = loss_topo * targets_batch

        length = img_embs.size(0)
        relation_img, relation_cap = self.similarity_model.sim_enc.forward_relation(img_embs, cap_embs, lengths)
        
        relation_i2t = self.forward_relation(relation_img, relation_cap, sims, length=batch_length, mode="v")
        relation_t2i = self.forward_relation(relation_img, relation_cap, sims.t(), length=batch_length, mode="t")

        relation_loss = (relation_i2t + relation_t2i) / 2
        relation_loss = relation_loss / batch_length
        relation_loss = relation_loss.mean()
        
        # glo_img, glo_cap = self.similarity_model.sim_enc.forward_emb_glo(img_embs, cap_embs, cap_lens)
        # 
        # sims_img = self.forward_sim_pair(glo_img, glo_img)
        # sims_cap = self.forward_sim_pair(glo_cap, glo_cap)
        # sims_v2t = self.forward_sim_pair(glo_img, glo_cap)
        # sims_t2v = self.forward_sim_pair(glo_cap, glo_img)

        # relation_v2t = self.forward_relation(sims_img, sims_cap, sims_v2t, length, "v")
        # relation_t2v = self.forward_relation(sims_img, sims_cap, sims_t2v, length, "t")

        # relation_loss = (relation_v2t + relation_t2v) / 2 * targets_batch
        # relation_loss = relation_loss.mean()

        # loss_topo_v2t, _ = self.info_nce_loss(sims_v2t, temp=1)
        # loss_topo_t2v, _ = self.info_nce_loss(sims_t2v, temp=1)
        # 
        # loss_topo = (loss_topo_v2t + loss_topo_t2v) / 2

        # self.similarity_model.topos[ids] = torch.cosine_similarity(sims_img, sims_cap, dim=1).data.detach().float()
        # self.similarity_model.topos[ids] = loss_topo.data.detach().float()
        self.similarity_model.preds[ids] = preds.data.detach().float()
        target_clean = (targets_batch * corrs).sum() / corrs.sum()
        target_noise = (targets_batch * (1 - corrs)).sum() / (1 - corrs).sum()

        loss = is_pseudo * self.opt.xi * loss_rank.mean() + is_noisy * is_lambda * relation_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss', loss.item(), batch_length)
        self.logger.update('Loss_rank', loss_rank.item(), batch_length)
        self.logger.update('Loss_relation', relation_loss.item(), batch_length)


def Relation_extraction(query, context, smooth, eps=1e-8, lens=None, attn_mask=None, attn_mode="vv"):
    queryT = torch.transpose(query, 1, 2)

    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    attn = torch.transpose(attn, 1, 2).contiguous()

    if attn_mask is not None:
        if attn_mode == "tt":
            attn_mask = attn_mask.unsqueeze(2)
            mask = torch.bmm(attn_mask, torch.transpose(attn_mask, 1, 2))
        if attn_mode == "vt":
            vv_mask = torch.ones((attn_mask.size(0), query.size(1))).cuda()
            vv_mask = vv_mask.unsqueeze(2)
            attn_mask = attn_mask.unsqueeze(2)
            mask = torch.bmm(vv_mask, torch.transpose(attn_mask, 1, 2))
        if attn_mode == "tv":
            vv_mask = torch.ones((attn_mask.size(0), context.size(1))).cuda()
            vv_mask = vv_mask.unsqueeze(2)
            attn_mask = attn_mask.unsqueeze(2)
            mask = torch.bmm(attn_mask, torch.transpose(vv_mask, 1, 2))

        attn = attn.masked_fill(mask==0, float("-inf"))

    if lens is not None:
        for i in range(len(lens)):
            attn[i, lens[i]:] = -1e9

    attn = F.softmax(attn * smooth, dim=2)

    attnT = torch.transpose(attn, 1, 2).contiguous()

    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return attn, weightedContext


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return x
