import torch
from torch import nn
import torch.nn.functional as F
from Networks.dcn_pack import DCNv2

##### Deformable Convolution Networks #####
class AlignNetwork(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(AlignNetwork, self).__init__()
        self.cal_offset = nn.Sequential(*[
            nn.Conv2d(2 * inChannels, inChannels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                        )
        self.DCN = DCNv2(inChannels, outChannels, 3, padding=1)

    def forward(self, Fi, Fe):
        offset = self.cal_offset(torch.cat((Fi, Fe), 1))
        offset_out, aligned_e = self.DCN(Fe, offset)
        Ff = torch.cat((Fi, aligned_e), 1)
        return offset_out, Ff


class Flow_Net_MVSEC_dc(nn.Module):
    def __init__(self, inChannels_i, inChannels_e):
        super(Flow_Net_MVSEC_dc, self).__init__()
        n_feat = 32
        ## SFE
        self.sfe_i = SFE(inChannels_i, n_feat)
        self.sfe_e = SFE(inChannels_e, n_feat)
        ## Image Encoder
        self.encoder_i0 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_i1 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_i2 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        ## Event Encoder
        self.encoder_e0 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_e1 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_e2 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        ## Deformable Convolution Networks
        self.dcn_0 = AlignNetwork(n_feat, n_feat)
        self.dcn_1 = AlignNetwork(2 * n_feat, 2 * n_feat)
        self.dcn_2 = AlignNetwork(4 * n_feat, 4 * n_feat)
        self.dcn_3 = AlignNetwork(8 * n_feat, 8 * n_feat)
        ## Flow Decoder
        self.decoder_f2 = Flow_Decoder_layer(16 * n_feat, 8 * n_feat, 8 * n_feat, scale=2)
        self.decoder_f1 = Flow_Decoder_layer(8 * n_feat, 4 * n_feat, 4 * n_feat, scale=1)
        self.decoder_f0 = Flow_Decoder_layer(4 * n_feat, 2 * n_feat, 2 * n_feat, scale=0)
        ## Disp Estimate
        self.sfe_o3 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 8 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o2 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 4 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o1 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 2 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o0 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 1 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.decoder_d2 = Disp_Decoder_layer(8 * n_feat, 8 * n_feat, 4 * n_feat, scale=2)
        self.decoder_d1 = Disp_Decoder_layer(8 * n_feat, 4 * n_feat, 2 * n_feat, scale=1)
        self.decoder_d0 = Disp_Decoder_layer(4 * n_feat, 2 * n_feat, 1 * n_feat, scale=0)

    def forward(self, I, E):
        ## SFE
        FI0 = self.sfe_i(I)
        FE0 = self.sfe_e(E)
        ## Encoder
        FI1 = self.encoder_i0(FI0)
        FI2 = self.encoder_i1(FI1)
        FI3 = self.encoder_i2(FI2)
        FE1 = self.encoder_e0(FE0)
        FE2 = self.encoder_e1(FE1)
        FE3 = self.encoder_e2(FE2)
        ## DCN align
        offset0, f0 = self.dcn_0(FI0, FE0)
        offset1, f1 = self.dcn_1(FI1, FE1)
        offset2, f2 = self.dcn_2(FI2, FE2)
        offset3, f3 = self.dcn_3(FI3, FE3)
        ## Flow Decoder
        x = f3
        x, flow2 = self.decoder_f2(x, f2)
        x, flow1 = self.decoder_f1(x, f1)
        x, flow0 = self.decoder_f0(x, f0)
        flow0_2 = flow2[:, :2, ...]
        flow1_2 = flow2[:, 2:, ...]
        flow0_1 = flow1[:, :2, ...]
        flow1_1 = flow1[:, 2:, ...]
        flow0_0 = flow0[:, :2, ...]
        flow1_0 = flow0[:, 2:, ...]
        # Disp Decoder
        x = self.sfe_o3(offset3)
        offset2 = self.sfe_o2(offset2)
        offset1 = self.sfe_o1(offset1)
        offset0 = self.sfe_o0(offset0)
        x, _ = self.decoder_d2(x, offset2)
        x, _ = self.decoder_d1(x, offset1)
        x, disp0 = self.decoder_d0(x, offset0)
        return [flow0_0, flow0_1, flow0_2], [flow1_0, flow1_1, flow1_2], disp0


class Flow_Net_DSEC_dc(nn.Module):
    def __init__(self, inChannels_i, inChannels_e):
        super(Flow_Net_DSEC_dc, self).__init__()
        n_feat = 32
        ## SFE
        self.sfe_i = SFE(inChannels_i, n_feat)
        self.sfe_e = SFE(inChannels_e, n_feat)
        ## Image Encoder
        self.encoder_i0 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_i1 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_i2 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        ## Event Encoder
        self.encoder_e0 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_e1 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_e2 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        ## Deformable Convolution Networks
        self.dcn_0 = AlignNetwork(n_feat, n_feat)
        self.dcn_1 = AlignNetwork(2 * n_feat, 2 * n_feat)
        self.dcn_2 = AlignNetwork(4 * n_feat, 4 * n_feat)
        self.dcn_3 = AlignNetwork(8 * n_feat, 8 * n_feat)
        ## Flow Decoder
        self.decoder_f2 = Flow_Decoder_layer(16 * n_feat, 8 * n_feat, 8 * n_feat, scale=2)
        self.decoder_f1 = Flow_Decoder_layer(8 * n_feat, 4 * n_feat, 4 * n_feat, scale=1)
        self.decoder_f0 = Flow_Decoder_layer(4 * n_feat, 2 * n_feat, 2 * n_feat, scale=0)
        ## Disp Estimate
        self.sfe_o3 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 8 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o2 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 4 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o1 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 2 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.sfe_o0 = nn.Sequential(*[nn.Conv2d(18, 2 * n_feat, kernel_size=5, padding=2, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(2 * n_feat, 1 * n_feat, kernel_size=3, padding=1, stride=1),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                    )
        self.decoder_d2 = Disp_Decoder_layer_DSEC(8 * n_feat, 8 * n_feat, 4 * n_feat, scale=2)
        self.decoder_d1 = Disp_Decoder_layer_DSEC(8 * n_feat, 4 * n_feat, 2 * n_feat, scale=1)
        self.decoder_d0 = Disp_Decoder_layer_DSEC(4 * n_feat, 2 * n_feat, 1 * n_feat, scale=0)

    def forward(self, I, E):
        ## SFE
        FI0 = self.sfe_i(I)
        FE0 = self.sfe_e(E)
        ## Encoder
        FI1 = self.encoder_i0(FI0)
        FI2 = self.encoder_i1(FI1)
        FI3 = self.encoder_i2(FI2)
        FE1 = self.encoder_e0(FE0)
        FE2 = self.encoder_e1(FE1)
        FE3 = self.encoder_e2(FE2)
        ## DCN align
        offset0, f0 = self.dcn_0(FI0, FE0)
        offset1, f1 = self.dcn_1(FI1, FE1)
        offset2, f2 = self.dcn_2(FI2, FE2)
        offset3, f3 = self.dcn_3(FI3, FE3)
        ## Flow Decoder
        x = f3
        x, flow2 = self.decoder_f2(x, f2)
        x, flow1 = self.decoder_f1(x, f1)
        x, flow0 = self.decoder_f0(x, f0)
        flow0_2 = flow2[:, :2, ...]
        flow1_2 = flow2[:, 2:, ...]
        flow0_1 = flow1[:, :2, ...]
        flow1_1 = flow1[:, 2:, ...]
        flow0_0 = flow0[:, :2, ...]
        flow1_0 = flow0[:, 2:, ...]
        # Disp Decoder
        x = self.sfe_o3(offset3)
        offset2 = self.sfe_o2(offset2)
        offset1 = self.sfe_o1(offset1)
        offset0 = self.sfe_o0(offset0)
        x, _ = self.decoder_d2(x, offset2)
        x, _ = self.decoder_d1(x, offset1)
        x, disp0 = self.decoder_d0(x, offset0)
        return [flow0_0, flow0_1, flow0_2], [flow1_0, flow1_1, flow1_2], disp0


class SynNet(nn.Module):
    def __init__(self, inChannels):
        super(SynNet, self).__init__()
        n_feat = 16
        self.sfe = SFE(inChannels, n_feat)
        self.encoder_layer0 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_layer1 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_layer2 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        self.resblock = Cascade_resnet_blocks(8 * n_feat, 2)
        self.decocer_layer2 = Image_Decoder_layer(8 * n_feat, 4 * n_feat, 4 * n_feat)
        self.decocer_layer1 = Image_Decoder_layer(4 * n_feat, 2 * n_feat, 2 * n_feat)
        self.decocer_layer0 = Image_Decoder_layer(2 * n_feat, n_feat, n_feat)

    def forward(self, x):
        f0 = self.sfe(x)
        f1 = self.encoder_layer0(f0)
        f2 = self.encoder_layer1(f1)
        f3 = self.encoder_layer2(f2)
        f = self.resblock(f3)
        f, syn_2 = self.decocer_layer2(f, f2)
        f, syn_1 = self.decocer_layer1(f, f1)
        f, syn_0 = self.decocer_layer0(f, f0)
        return syn_0


class UNet(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UNet, self).__init__()
        n_feat = 16
        self.sfe = SFE(inChannels, n_feat)
        self.encoder_layer1 = Encoder_layer(n_feat, 2 * n_feat, 5)
        self.encoder_layer2 = Encoder_layer(2 * n_feat, 4 * n_feat, 3)
        self.encoder_layer3 = Encoder_layer(4 * n_feat, 8 * n_feat, 3)
        self.resblock = Cascade_resnet_blocks(8 * n_feat, 2)
        self.decocer_layer1 = Decoder_layer(8 * n_feat, 4 * n_feat, 4 * n_feat)
        self.decocer_layer2 = Decoder_layer(4 * n_feat, 2 * n_feat, 2 * n_feat)
        self.decocer_layer3 = Decoder_layer(2 * n_feat, n_feat, n_feat)
        self.conv = conv2d(in_planes=n_feat, out_planes=outChannels, batch_norm=False, activation=nn.Tanh(),
                           kernel_size=3, stride=1)

    def forward(self, x):
        f0 = self.sfe(x)
        f1 = self.encoder_layer1(f0)
        f2 = self.encoder_layer2(f1)
        f3 = self.encoder_layer3(f2)
        f = self.resblock(f3)
        f = self.decocer_layer1(f, f2)
        f = self.decocer_layer2(f, f1)
        f = self.decocer_layer3(f, f0)
        out = self.conv(f)
        return out

##### RDN
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, out_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # # up-sampling
        # assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.output = nn.Conv2d(self.G0, out_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x)
        return x


##### modules
class SFE(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SFE, self).__init__()
        self.sfe = nn.Sequential(*[nn.Conv2d(inChannels, outChannels, kernel_size=5, padding=2, stride=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1, stride=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True)]
                                 )

    def forward(self, x):
        x = self.sfe(x)
        return x

class Encoder_layer(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size):
        super(Encoder_layer, self).__init__()
        self.downsample = down(inChannels, outChannels, filterSize=kernel_size)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)

    def forward(self, x):
        x = self.downsample(x)
        x = self.resblocks(x)
        return x

class skip_connection_cat(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels):
        super(skip_connection_cat, self).__init__()
        self.conv1 = conv2d(in_planes=inChannels+preChannels, out_planes=outChannels, batch_norm=False, activation=nn.LeakyReLU(),
                            kernel_size=5, stride=1)
        self.conv2 = conv2d(in_planes=outChannels, out_planes=outChannels, batch_norm=False,
                            activation=nn.LeakyReLU(), kernel_size=3, stride=1)

    def forward(self, x, f):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        H, W = f.shape[2], f.shape[3]
        c = torch.cat((x[:, :, :H, :W], f), 1)
        c = self.conv1(c)
        c = self.conv2(c)
        return c

class Decoder_layer(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels):
        super(Decoder_layer, self).__init__()
        self.sc = skip_connection_cat(inChannels, outChannels, preChannels)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)
        self.conv = conv2d(in_planes=outChannels, out_planes=outChannels, batch_norm=False, activation=nn.LeakyReLU(),
                           kernel_size=3, stride=1)

    def forward(self, x, f):
        x = self.sc(x, f)
        x = self.resblocks(x)
        x = self.conv(x)
        return x

class Image_Decoder_layer(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels):
        super(Image_Decoder_layer, self).__init__()
        self.sc = skip_connection_cat(inChannels, outChannels, preChannels)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)
        self.pred = ImagePred(outChannels)

    def forward(self, x, f):
        x = self.sc(x, f)
        x = self.resblocks(x)
        x, image = self.pred(x)
        return x, image

class ChannelAttention(nn.Module):
    ## channel attention block
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    ## spatial attention block
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = conv2d(in_planes=inChannels, out_planes=outChannels, batch_norm=False, activation=nn.LeakyReLU(),
                            kernel_size=filterSize, stride=1)
        self.conv2 = conv2d(in_planes=outChannels, out_planes=outChannels, batch_norm=False, activation=nn.LeakyReLU(),
                            kernel_size=filterSize, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ImagePred(nn.Module):
    def __init__(self, inChannels):
        super(ImagePred, self).__init__()
        self.conv1 = conv2d(in_planes=inChannels, out_planes=inChannels, batch_norm=False, activation=nn.LeakyReLU(),
                            kernel_size=3, stride=1)
        self.ca = ChannelAttention(inChannels)
        self.sa = SpatialAttention()
        self.conv2 = conv2d(in_planes=inChannels, out_planes=3, batch_norm=False, activation=nn.Sigmoid(),
                            kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        pred = self.conv2(x)
        return x, pred

class PredDisp(nn.Module):
    def __init__(self, inChannels, scale):
        super(PredDisp, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1)
        self.acf1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.ca = ChannelAttention(inChannels)
        self.sa = SpatialAttention()
        self.conv2 = nn.Conv2d(inChannels, 1, kernel_size=3, padding=1, stride=1)
        self.acf2 = nn.Sigmoid()
        self.scale = scale

    def forward(self, x):
        x = self.acf1(self.conv1(x))
        x = self.ca(x) * x
        x = self.sa(x) * x
        pred = self.acf2(self.conv2(x)) * 40. / 2**self.scale
        return x, pred

class PredDisp_DSEC(nn.Module):
    def __init__(self, inChannels, scale):
        super(PredDisp_DSEC, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1)
        self.acf1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.ca = ChannelAttention(inChannels)
        self.sa = SpatialAttention()
        self.conv2 = nn.Conv2d(inChannels, 1, kernel_size=3, padding=1, stride=1)
        self.acf2 = nn.Sigmoid()
        self.scale = scale

    def forward(self, x):
        x = self.acf1(self.conv1(x))
        x = self.ca(x) * x
        x = self.sa(x) * x
        pred = self.acf2(self.conv2(x)) * 80. / 2**self.scale
        return x, pred

class PredFlow(nn.Module):
    def __init__(self, inChannels, scale):
        super(PredFlow, self).__init__()
        self.conv1 = conv2d(in_planes=inChannels, out_planes=inChannels, batch_norm=False, activation=nn.LeakyReLU(),
                            kernel_size=3, stride=1)
        self.ca = ChannelAttention(inChannels)
        self.sa = SpatialAttention()
        self.conv2 = conv2d(in_planes=inChannels, out_planes=4, batch_norm=False, activation=nn.Tanh(), kernel_size=3,
                            stride=1)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        flow = self.conv2(x) * 128. / 2**self.scale
        return x, flow

class Flow_Decoder_layer(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels, scale):
        super(Flow_Decoder_layer, self).__init__()
        self.sc = skip_connection_cat(inChannels, outChannels, preChannels)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)
        self.pred = PredFlow(outChannels, scale)

    def forward(self, x, f):
        x = self.sc(x, f)
        x = self.resblocks(x)
        x, flow = self.pred(x)
        return x, flow

class Disp_Decoder_layer(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels, scale=0):
        super(Disp_Decoder_layer, self).__init__()
        self.sc = skip_connection_cat(inChannels, outChannels, preChannels)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)
        self.pred = PredDisp(outChannels, scale)

    def forward(self, x, f):
        x = self.sc(x, f)
        x = self.resblocks(x)
        x, pred = self.pred(x)
        return x, pred

class Disp_Decoder_layer_DSEC(nn.Module):
    def __init__(self, inChannels, outChannels, preChannels, scale=0):
        super(Disp_Decoder_layer_DSEC, self).__init__()
        self.sc = skip_connection_cat(inChannels, outChannels, preChannels)
        self.resblocks = Cascade_resnet_blocks(in_planes=outChannels, n_blocks=2)
        self.pred = PredDisp_DSEC(outChannels, scale)

    def forward(self, x, f):
        x = self.sc(x, f)
        x = self.resblocks(x)
        x, pred = self.pred(x)
        return x, pred

class conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, batch_norm, activation, kernel_size=3, stride=1):
        super(conv2d, self).__init__()

        use_bias = True
        if batch_norm:
            use_bias = False

        modules = []
        modules.append(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=use_bias))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_planes))
        if activation:
            modules.append(activation)

        self.net = nn.Sequential(*modules)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_planes):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_planes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_conv_block(self, in_planes):
        conv_block = []
        conv_block += [
            conv2d(in_planes=in_planes, out_planes=in_planes, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                   stride=1)]
        conv_block += [
            conv2d(in_planes=in_planes, out_planes=in_planes, batch_norm=False, activation=False, kernel_size=3,
                   stride=1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Cascade_resnet_blocks(nn.Module):
    def __init__(self, in_planes, n_blocks):
        super(Cascade_resnet_blocks, self).__init__()

        resnet_blocks = []
        for i in range(n_blocks):
            resnet_blocks += [ResnetBlock(in_planes)]

        self.net = nn.Sequential(*resnet_blocks)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
