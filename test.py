import sys
import argparse
from Networks.SEVFI import *
from script.dataloader import *
sys.path.append('../Networks')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def test_STFI(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load SEVFI model
    if opt.dataset =='SEID':
        net = SEVFI_dc_SEID()
    elif opt.dataset =='DSEC':
        net = SEVFI_dc_DSEC()
    elif opt.dataset =='MVSEC':
        net = SEVFI_dc_MVSEC()
    else:
        raise ValueError(f'Unknown dataset: {opt.dataset}')
    model_path = opt.model_path + opt.dataset + '.pth'
    data_path = opt.origin_path + opt.dataset
    net = torch.nn.DataParallel(net)
    print("Testing STFI model: " + model_path)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.to(device)
    net = net.eval()
    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters)
    print('skip=insert: %d' % opt.num_insert)
    # prepare dataset
    for i in range(len(opt.test_list)):
        test_path = os.path.join(data_path, opt.test_list[i])
        if opt.dataset == 'SEID':
            testDataset = test_SEID_sevfi(data_path=test_path, num_bins=15, num_skip=opt.num_skip,
                                          num_insert=opt.num_insert)
        elif opt.dataset == 'DSEC':
            testDataset = test_DSEC_sevfi(data_path=test_path, num_bins=15, num_skip=opt.num_skip,
                                          num_insert=opt.num_insert)
        elif opt.dataset == 'MVSEC':
            testDataset = test_MVSEC_sevfi(data_path=test_path, num_bins=15, num_skip=opt.num_skip,
                                           num_insert=opt.num_insert)
        result_path = opt.save_path + opt.dataset + '/insert_' + str(opt.num_insert) + '/' + opt.test_list[i] + '/'
        utils.mkdir(result_path)
        disp_path = opt.save_path + opt.dataset + '/disp_' + str(opt.num_insert) + '/' + opt.test_list[i] + '/'
        utils.mkdir(disp_path)
        # testing
        print(opt.test_list[i])
        print('%d / %d' % (i+1, len(opt.test_list)))
        with torch.no_grad():
            for k in range(len(testDataset)):
                sample = testDataset[k]
                print("Processing img %d ..." % (k))
                # load data
                image_0 = sample['image_0']
                image_1 = sample['image_1']
                eframes_t1 = sample['eframes_t1']
                eframes_t0 = sample['eframes_t0']
                iwe = sample['iwe']
                weight = sample['weight']

                # CPU to GPU
                B = 1
                N, C, H, W = eframes_t0.shape
                image_0 = torch.from_numpy(image_0).permute(2, 0, 1)
                image_1 = torch.from_numpy(image_1).permute(2, 0, 1)
                image_0 = image_0.unsqueeze(1).repeat(N, 1, 1, 1)
                image_1 = image_1.unsqueeze(1).repeat(N, 1, 1, 1)
                image_0 = image_0.reshape(B * N, 3, H, W).float().to(device)
                image_1 = image_1.reshape(B * N, 3, H, W).float().to(device)
                eframes_t1 = torch.from_numpy(eframes_t1).reshape(B * N, C, H, W).float().to(device)
                eframes_t0 = torch.from_numpy(eframes_t0).reshape(B * N, C, H, W).float().to(device)
                iwe = torch.from_numpy(iwe).reshape(B * N, 1, H, W).float().to(device)
                weight = torch.from_numpy(weight).reshape(B * N).float().to(device)

                # process by SEVFI network
                image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1 = net(image_0, image_1, eframes_t0,
                                                                                         eframes_t1, iwe, weight)
                final_t = torch.clamp(image_final, min=0, max=1)
                output = final_t.reshape(B, N, 3, H, W) * 255.
                output_disp = disp.reshape(B, N, 1, H, W)

                # save interpolated images
                save_image_0 = image_0[0, :].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                name_0 = result_path + '{:05d}'.format(int(k * (opt.num_insert + 1))) + '.png'
                cv2.imwrite(name_0, save_image_0)
                save_image_1 = image_1[0, :].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                name_1 = result_path + '{:05d}'.format(int((k + 1) * (opt.num_insert + 1))) + '.png'
                cv2.imwrite(name_1, save_image_1)
                for i in range(opt.num_insert):
                    # save images
                    output_image = output[0, i, :].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                    out_name = result_path + '{:05d}'.format(int(k * (opt.num_insert + 1) + i + 1)) + '.png'
                    cv2.imwrite(out_name, output_image)
                    # save disparities
                    out_disp = output_disp[0, i, :].squeeze().cpu().detach().numpy()
                    disp_name = disp_path + '{:05d}'.format(int(k * (opt.num_insert + 1) + i + 1)) + '.png'
                    cv2.imwrite(disp_name, out_disp)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser(description="Test SEVFI")
    parser.add_argument("--dataset", type=str, default="SEID", help="dataset name")
    parser.add_argument("--model_path", type=str, default="./PreTrained/", help="path of pretrained model")
    parser.add_argument("--origin_path", type=str, default="./sample/dataset/", help="path of test datasets")
    parser.add_argument("--test_list", type=list, default=[], help="list of test name")
    parser.add_argument("--save_path", type=str, default="./sample/result/", help="saving path")
    parser.add_argument("--num_skip", type=int, default=5, help="num of skip frames")
    parser.add_argument("--num_insert", type=int, default=5, help="num of insert frames")

    opt = parser.parse_args()

    test_filepath = opt.origin_path + opt.dataset + '/'
    test_name = os.listdir(test_filepath)
    for iter in test_name:
        opt.test_list.append(iter)
    test_STFI(opt)
