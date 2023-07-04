from script import utils
from Networks.modules import *


class SEVFI_dc_MVSEC(nn.Module):
    def __init__(self, ):
        super(SEVFI_dc_MVSEC, self).__init__()
        self.flow_net = Flow_Net_MVSEC_dc(inChannels_i=6, inChannels_e=30)
        self.syn_net = SynNet(inChannels=7)
        self.mask_net = UNet(inChannels=14, outChannels=3)
        self.refine = RDN(scale_factor=1, num_channels=9, out_channels=3, num_features=16, growth_rate=16, num_blocks=8,
                          num_layers=4)

    def forward(self, image_0, image_1, eframes_t0, eframes_t1, iwe, weight):
        image_0_nor = utils.normalization(image_0)
        image_1_nor = utils.normalization(image_1)
        # disp estimation
        inp_i = torch.cat((image_0_nor, image_1_nor), 1)
        inp_e = torch.cat((eframes_t0, eframes_t1), 1)
        # flow_right estimation
        flowlist_t0, flowlist_t1, disp = self.flow_net(inp_i, inp_e)
        flow_t0 = flowlist_t0[0]
        flow_t1 = flowlist_t1[0]
        # get Image_warped_t
        warped_0t_nor = utils.warp_images_with_flow(image_0_nor, flow_t0)
        warped_1t_nor = utils.warp_images_with_flow(image_1_nor, flow_t1)
        # weight image
        wei = torch.ones_like(warped_0t_nor)
        for l in range(len(weight)):
            wei[l] = weight[l] * wei[l]
        wei = wei[:, 0, ...].unsqueeze(1)
        # warp iwe
        iwe_warped, _ = utils.disp_warp(iwe, disp)
        # SynNet
        inp_s = torch.cat((image_0_nor, image_1_nor, iwe_warped), 1)
        image_syn = self.syn_net(inp_s)
        # FusionNet
        inp_m = torch.cat((warped_0t_nor, warped_1t_nor, image_syn, flow_t0, flow_t1, wei), 1)
        mask = self.mask_net(inp_m)
        mask = F.softmax(mask, dim=1)
        image_fuse = mask[:, 0, ...].unsqueeze(1) * warped_0t_nor + mask[:, 1, ...].unsqueeze(1) * warped_1t_nor + \
                     mask[:, 2, ...].unsqueeze(1) * image_syn
        # RefineNet
        inp_r = torch.cat((warped_0t_nor, warped_1t_nor, image_syn), 1)
        res = self.refine(inp_r)
        image_final = image_fuse + res
        return image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1

class SEVFI_dc_DSEC(nn.Module):
    def __init__(self, ):
        super(SEVFI_dc_DSEC, self).__init__()
        self.flow_net = Flow_Net_DSEC_dc(inChannels_i=6, inChannels_e=30)
        self.syn_net = SynNet(inChannels=7)
        self.mask_net = UNet(inChannels=14, outChannels=3)
        self.refine = RDN(scale_factor=1, num_channels=9, out_channels=3, num_features=16, growth_rate=16, num_blocks=8,
                          num_layers=4)

    def forward(self, image_0, image_1, eframes_t0, eframes_t1, iwe, weight):
        image_0_nor = utils.normalization(image_0)
        image_1_nor = utils.normalization(image_1)
        # disp estimation
        inp_i = torch.cat((image_0_nor, image_1_nor), 1)
        inp_e = torch.cat((eframes_t0, eframes_t1), 1)
        # flow_right estimation
        flowlist_t0, flowlist_t1, disp = self.flow_net(inp_i, inp_e)
        flow_t0 = flowlist_t0[0]
        flow_t1 = flowlist_t1[0]
        # get Image_warped_t
        warped_0t_nor = utils.warp_images_with_flow(image_0_nor, flow_t0)
        warped_1t_nor = utils.warp_images_with_flow(image_1_nor, flow_t1)
        # weight image
        wei = torch.ones_like(warped_0t_nor)
        for l in range(len(weight)):
            wei[l] = weight[l] * wei[l]
        wei = wei[:, 0, ...].unsqueeze(1)
        # warp iwe
        iwe_warped, _ = utils.disp_warp(iwe, disp)
        # SynNet
        inp_s = torch.cat((image_0_nor, image_1_nor, iwe_warped), 1)
        image_syn = self.syn_net(inp_s)
        # FusionNet
        inp_m = torch.cat((warped_0t_nor, warped_1t_nor, image_syn, flow_t0, flow_t1, wei), 1)
        mask = self.mask_net(inp_m)
        mask = F.softmax(mask, dim=1)
        image_fuse = mask[:, 0, ...].unsqueeze(1) * warped_0t_nor + mask[:, 1, ...].unsqueeze(1) * warped_1t_nor + \
                     mask[:, 2, ...].unsqueeze(1) * image_syn
        # RefineNet
        inp_r = torch.cat((warped_0t_nor, warped_1t_nor, image_syn), 1)
        res = self.refine(inp_r)
        image_final = image_fuse + res
        return image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1

class SEVFI_dc_SEID(nn.Module):
    def __init__(self, ):
        super(SEVFI_dc_SEID, self).__init__()
        self.flow_net = Flow_Net_MVSEC_dc(inChannels_i=6, inChannels_e=30)
        self.syn_net = SynNet(inChannels=7)
        self.mask_net = UNet(inChannels=14, outChannels=3)
        self.refine = RDN(scale_factor=1, num_channels=9, out_channels=3, num_features=16, growth_rate=16, num_blocks=8,
                          num_layers=4)

    def forward(self, image_0, image_1, eframes_t0, eframes_t1, iwe, weight):
        image_0_nor = utils.normalization(image_0)
        image_1_nor = utils.normalization(image_1)
        # disp estimation
        inp_i = torch.cat((image_0_nor, image_1_nor), 1)
        inp_e = torch.cat((eframes_t0, eframes_t1), 1)
        # flow_right estimation
        flowlist_t0, flowlist_t1, disp = self.flow_net(inp_i, inp_e)
        flow_t0 = flowlist_t0[0]
        flow_t1 = flowlist_t1[0]
        # get Image_warped_t
        warped_0t_nor = utils.warp_images_with_flow(image_0_nor, flow_t0)
        warped_1t_nor = utils.warp_images_with_flow(image_1_nor, flow_t1)
        # weight image
        wei = torch.ones_like(warped_0t_nor)
        for l in range(len(weight)):
            wei[l] = weight[l] * wei[l]
        wei = wei[:, 0, ...].unsqueeze(1)
        # warp iwe
        iwe_warped, _ = utils.disp_warp(iwe, disp)
        # SynNet
        inp_s = torch.cat((image_0_nor, image_1_nor, iwe_warped), 1)
        image_syn = self.syn_net(inp_s)
        # FusionNet
        inp_m = torch.cat((warped_0t_nor, warped_1t_nor, image_syn, flow_t0, flow_t1, wei), 1)
        mask = self.mask_net(inp_m)
        mask = F.softmax(mask, dim=1)
        image_fuse = mask[:, 0, ...].unsqueeze(1) * warped_0t_nor + mask[:, 1, ...].unsqueeze(1) * warped_1t_nor + \
                     mask[:, 2, ...].unsqueeze(1) * image_syn
        # RefineNet
        inp_r = torch.cat((warped_0t_nor, warped_1t_nor, image_syn), 1)
        res = self.refine(inp_r)
        image_final = image_fuse + res
        return image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1