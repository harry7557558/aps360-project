import torch


class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_a, lambda_p, lambda_f, lambda_t, lambda_c, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.lambda_a = lambda_a  # adversarial
        self.lambda_p = lambda_p  # pixel loss
        self.lambda_f = lambda_f  # feature loss (perceptual)
        self.lambda_t = lambda_t  # texture matching loss
        self.lambda_c = lambda_c  # correct color shift caused by GAN

        # loss function for GAN
        self.bce = torch.nn.BCEWithLogitsLoss()

        # VGG16 model for feature and texture loss
        import torchvision.models
        vgg16 = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # vgg first layer
        self.feature_extractor = vgg16.features[:4].to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor_1 = lambda x: self.feature_extractor((x-vgg_mean)/vgg_std)
        # print(self.feature_extractor)

        # vgg second layer, after first layer
        self.feature_extractor_2 = vgg16.features[4:9].to(device)
        for param in self.feature_extractor_2.parameters():
            param.requires_grad = False
        # print(self.feature_extractor_2)
    
    # @staticmethod
    # def gram(x, s=None):
    #     # g = torch.einsum('kaij,kbij->kab', x, x)
    #     x1 = x.reshape((x.shape[0], x.shape[1], -1))
    #     g = torch.matmul(x1, x1.transpose(2, 1))
    #     return g / (x.shape[2]*x.shape[3])
            
    @staticmethod
    def gram(x, s):
        if x.shape[2] % s != 0 or x.shape[3] % s != 0:
            raise ValueError("Image dimension not multiple of texture patch size")
        x = x.view(x.shape[0], x.shape[1], x.shape[2]//s, s, x.shape[3]//s, s)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1, s*s)
        g = torch.matmul(x, x.transpose(3, 2))
        return g / (s*s)
    
    def forward(self, generated, target, discriminator_output, is_discriminator=False):
        # normalization parameters
        normalize_l2 = 1.0 / 0.05  # change this to estimated L1 loss
        normalize_vgg = 1.0 / 0.225
        normalize_l2_vgg = normalize_l2 / normalize_vgg
        normalize_l2_vgg_gram = normalize_l2_vgg ** 2

        # adversarial Loss
        if is_discriminator:
            return self.bce(discriminator_output, target)
        loss_a = -torch.mean(discriminator_output)

        # pixel loss (L1+L2)
        loss_p2 = torch.mean((generated-target)**2) * normalize_l2
        loss_p1 = torch.mean(torch.abs(generated-target))
        loss_p = 0.5*(loss_p1+loss_p2)

        # feature loss (L1+L2)
        gen_features = self.feature_extractor_1(generated)
        tgt_features = self.feature_extractor_1(target)
        loss_f1l2 = torch.mean((gen_features-tgt_features)**2) * normalize_l2_vgg
        loss_f1l1 = torch.mean(torch.abs(gen_features-tgt_features))
        gen_features_2 = self.feature_extractor_2(gen_features)
        tgt_features_2 = self.feature_extractor_2(tgt_features)
        loss_f2l2 = torch.mean((gen_features_2-tgt_features_2)**2) * normalize_l2_vgg
        loss_f2l1 = torch.mean(torch.abs(gen_features_2-tgt_features_2))
        loss_f = 0.25*(loss_f1l2+loss_f1l1+loss_f2l2+loss_f2l1)

        # texture loss
        loss_t1 = torch.mean(self.gram(gen_features,16)-self.gram(tgt_features,16))**2
        # loss_t2 = torch.mean(self.gram(gen_features_2,8)-self.gram(tgt_features_2,8))**2
        loss_t2 = torch.mean(self.gram(gen_features_2,16)-self.gram(tgt_features_2,16))**2
        loss_t = (0.5*(loss_t1+loss_t2))**0.5 * normalize_l2_vgg_gram

        # color correction loss
        loss_cm = torch.mean(torch.mean(target-generated, axis=(2,3))**2) / normalize_l2
        loss_cd = torch.mean((torch.std(target,axis=(2,3))-torch.std(generated,axis=(2,3)))**2) / normalize_l2
        loss_c = loss_cm + loss_cd

        # composite loss
        total_loss = \
            self.lambda_a * loss_a + \
            self.lambda_p * loss_p + \
            self.lambda_f * loss_f + \
            self.lambda_t * loss_t + \
            self.lambda_c * loss_c
        # print(loss_a.item(), loss_p.item(), loss_f.item(), loss_t.item(), loss_c.item())

        return total_loss



if __name__ == "__main__":

    # make sure there is no error

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # training: set loss_a and loss_c to 0, wait for image to stabilize, then set it to 0.1, train again
    lossfun = CombinedLoss(0.0, 1.0, 1.0, 0.0, 4.0)
    lossfun = CombinedLoss(0.1, 1.0, 1.0, 1.0, 4.0, device=device)
    print(lossfun)

    # should give no error
    img1 = torch.randn((8, 3, 256, 256), device=device)
    img2 = img1 + 0.05 * torch.randn_like(img1)
    loss = lossfun(img1, img2, torch.randn((8,)), False)
    print(loss)
    loss = lossfun(img1, img1, torch.randn((8,)), False)
    print(loss)

    # should give an error
    img1 = torch.randn((8, 3, 254, 254), device=device)
    loss = lossfun(img1, img1, torch.randn((8,)), False)
