import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDatasetForSeg(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        # self.seg_colors = loadmat('data/color150.mat')['colors']
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        w4 = int(w / 4)
        A = AB.crop((0, 0, w4, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        if not self.opt.for_segB:
            seg = AB.crop((w2, 0, w2+w4, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        else:
            seg = AB.crop((w2+w4, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)


        if self.opt.generate_fake_b:
            B = AB.crop((w4, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        else:
            B = AB.crop((w2+w4, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        seg = transforms.ToTensor()(seg)

        # print("A: {}\nB:{}\nA_seg:{}".format(A, B, A_seg))

        # A_seg_filtered = []
        # for i in range(0, len(self.seg_colors)):
        #     print(A_seg_filtered[:,])
        #     A_seg_f = A_seg[:, :, A_seg == self.seg_colors[i]]
        #     A_seg_filtered.append(A_seg_f)

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        seg = seg[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        seg = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(seg)

        if self.opt.use_dist:
            dist = float(AB_path.split("/")[-1].split("_")[1])
            # print("path: {}, dist:{}".format(AB_path, dist))
            padding = torch.ones(1, self.opt.fineSize, self.opt.fineSize) * dist
            A = torch.cat((A, padding), 0)

        if self.opt.for_seg:
            A = torch.cat((A, seg), 0)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
