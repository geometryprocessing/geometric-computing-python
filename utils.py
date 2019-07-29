from __future__ import division

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

import igl
import yaml
from yaml import CLoader as Loader, CDumper as Dumper

import os
import os.path as osp
import glob
import yaml
import igl
import numpy as np
import torch
from torch_geometric.data import (Data, InMemoryDataset)
    
def read_model(obj_path, feat_path):
    m = {}
    m["vertices"], _, m["normals"], m["face_indices"], _, m["normal_indices"] = igl.read_obj(obj_path)
            
    with open(feat_path) as fi:
        m["features"] = yaml.load(fi, Loader=Loader)
    return m




class ABCDataset(InMemoryDataset):
    r""" The ABC dataset from the `"ABC: A Big CAD Model Dataset for Geometric Deep Learning"
    <https://deep-geometry.github.io/abc-dataset/>`_paper, containing about 1M CAD models
    with ground truth for surface normals, patch segmentation and sharp features.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The categories of the CAD
            model ground truth values. Can be one of 'Normals', 'Patches', 'Curves'.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deep-geometry.github.io/abc-dataset/'

    def __init__(self, root, typ="Curves", train=True, transform=None, pre_transform=None, pre_filter=None):
        super(ABCDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        if typ == "Curves":
            p = path
        if typ == "Normals":
            p = path.replace(".pt", "_normals.pt")
        self.data, self.slices = torch.load(p)
        from yaml import CLoader as Loader, CDumper as Dumper

    @property
    def raw_file_names(self):
        return [
            'train_data', 'train_label', 'test_data', 'test_label'
        ]

    @property
    def processed_file_names(self):
        #cats = '_'.join([cat[:3].lower() for cat in self.categories])
        fns = []
        for n in ["train", "test"]:
            for c in ['data']:
                fns.append('{}_{}.pt'.format(n, c))
        return fns
    
    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        data = self.data
        return data.y.max().item() + 1 if data.y.dim() == 1 else data.y.size(1)

#    def download(self):
#        for name in self.raw_file_names:
#            url = '{}/{}.zip'.format(self.url, name)
#            path = download_url(url, self.raw_dir)
#            extract_zip(path, self.raw_dir)
#            os.unlink(path)

    def process_raw_path(self, data_path, label_path):
        data_list = []
        obj_paths = sorted(glob.glob(osp.join(data_path, '*.obj')))
        feat_paths = sorted(glob.glob(osp.join(label_path, '*.yml')))
        
        points = []
        normals = []
        patches = []
        faces = []
        names = []
        types = []
        face_patches = []
        t_map = {"Plane": 0, "Cylinder": 1, "Cone": 2, "Sphere": 3, "Torus": 4, "Bezier": 5, "BSpline": 6, "Revolution": 7,"Extrusion": 8, "Other": 9}
        cnt = 0
        for idx, obj in enumerate(obj_paths[:250]):
            if cnt == 100:
                break
            print(idx, cnt)
            if os.path.getsize(obj) >= 10 * 1024**2 or os.path.getsize(feat_paths[idx+5]) >= 10 * 1024**2:
                #print("Skipping large file", obj)
                continue
            m = read_model(obj, feat_paths[idx])
            normal = cm.get_averaged_normals(m)
            #print(normal.shape)
            
            patch = np.zeros(m["vertices"].shape[0], dtype=np.long)
            typ = np.zeros(m["vertices"].shape[0], dtype=np.long)
            f_patch = np.zeros(m["face_indices"].shape[0], dtype=np.long)
            invalid = False
            for i, fe in enumerate(m["features"]["surfaces"]):
                if invalid:
                    break
                t = t_map[fe["type"]]
                for j in fe["vert_indices"]:
                    if j >= patch.shape[0]:
                        invalid = True
                        break
                    patch[j] = i
                    typ[j] = t
                for j in fe["face_indices"]:
                    f_patch[j] = i
                                  
            if invalid:
                #print("Skipping model %s"%obj)
                continue

            for i, fe in enumerate(m["features"]["curves"]):
                val = -1
                if fe["sharp"]:
                    val = -2
                    
                for j in range(len(fe["vert_indices"])):
                    v_s = fe["vert_indices"][j]
                    patch[v_s] = val
            points.append(torch.tensor(m["vertices"].astype(np.float32)).squeeze())
            normals.append(torch.tensor(normal.astype(np.float32)).squeeze())
            patches.append(torch.tensor(patch).squeeze())
            types.append(torch.tensor(typ).squeeze())
            faces.append(m["face_indices"])
            face_patches.append(f_patch)
            names.append(obj)
            cnt += 1

        cnt = 0
        for (v, n, p, t) in zip(points, normals, patches, types):
            cnts = torch.tensor(np.ones(p.shape, dtype=np.long)*cnt)
            cnt += 1
            data = Data(pos=v, idx=cnts, y_typ=t, y_patch=p, y_normal=n)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list, faces, names, face_patches

    def process(self):
        train_data, train_faces, train_names, train_fpatches = self.process_raw_path(*self.raw_paths[0:2])
        test_data, test_faces, test_names, test_fpatches = self.process_raw_path(*self.raw_paths[2:4])

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(test_data), self.processed_paths[1])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)





def accuracy(pred, target):
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()


def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)


def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)


def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)


def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)


def precision(pred, target, num_classes):
    r"""Computes the precision
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target, num_classes):
    r"""Computes the recall
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score


def intersection_and_union(pred, target, num_classes, batch=None):
    r"""Computes intersection and union of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    if batch is None:
        i = (pred & target).sum(dim=0)
        u = (pred | target).sum(dim=0)
    else:
        i = scatter_add(pred & target, batch, dim=0)
        u = scatter_add(pred | target, batch, dim=0)

    return i, u


def mean_iou(pred, target, num_classes, batch=None):
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: :class:`Tensor`
    """
    i, u = intersection_and_union(pred, target, num_classes, batch)
    iou = i.to(torch.float) / u.to(torch.float)
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


