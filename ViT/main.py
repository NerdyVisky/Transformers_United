import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Import and Version Check
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

img, label = train_data[0]
# print(img.shape)
# print(len(train_data.data))
# print(len(train_data.targets))
class_names = train_data.classes
# print(class_names)
new_class_names = []
for name in class_names:
    new_class_names.append(name[0])

# print(new_class_names)
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True
                              )
test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False
                             )
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
img_size = 28
height = 28
width = 28
channels = 3
patch_size = 7

number_of_patches = int((height * width) // patch_size ** 2)
# print(number_of_patches)
image_permuted = image.permute(1, 2, 0)
plt.figure(figsize=(patch_size, patch_size))
plt.imshow(image_permuted[:patch_size, :, :])
# plt.show()

assert height % patch_size == 0, "Invalid patch size."
fig, axs = plt.subplots(nrows=img_size // patch_size,  # need int not float
                        ncols=img_size // patch_size,
                        figsize=(number_of_patches, number_of_patches),
                        sharex=True,
                        sharey=True)

for i, patch_height in enumerate(range(0, img_size, patch_size)):
    for j, patch_width in enumerate(range(0, img_size, patch_size)):
        axs[i, j].imshow(image_permuted[patch_height:patch_height + patch_size,
                         patch_width:patch_width + patch_size,
                         :])
        axs[i, j].set_ylabel(i + 1,
                             rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        axs[i, j].set_xlabel(j + 1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

fig.suptitle(f"{new_class_names[label]} -> Patchified", fontsize=16)
# plt.show()

import random
random_indexes = random.sample(range(0, 49), k=5)
print(f"Showing random convolutional feature maps from indexes: {random_indexes}")
#
# # Create plot
# fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7, 7))
#
# # Plot random image feature maps
# for i, idx in enumerate(random_indexes):
#     image_conv_feature_map = embed_image[:, idx, :, :]
#     axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy())
#     axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     plt.show()
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=1,
                 patch_size:int=7,
                 embedding_dim:int=49):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3
        )

    def forward(self, x):
        img_size = x.shape[3]
        assert img_size % patch_size == 0, "Patch size not compatible with image size"
        out = self.patch_embed(x)
        out = self.flatten(out)
        return out.permute(0, 2, 1)

def perform_embedding(image):
    patch_img = PatchEmbedding(in_channels=1, patch_size=7, embedding_dim=49)
    patched_img = patch_img(image)
    # print(patched_img.shape)
    batch_size = patched_img.shape[0]
    embedding_dimension = patched_img.shape[-1]
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                               requires_grad=True)
    patch_embeds = torch.cat((class_token, patched_img), dim=1)
    # print(patch_embeds[:, :5, :3])
    # print(patch_embeds.shape)
    position_embeds = nn.Parameter(torch.ones(
        1, number_of_patches + 1, embedding_dimension
    ), requires_grad=True)
    patch_and_position_embeds = patch_embeds + position_embeds
    return patch_and_position_embeds


patch_and_position_embeds = perform_embedding(image.unsqueeze(0))
# print(patch_and_position_embeds[:, :5, :3])
# print(patch_and_position_embeds.shape)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim:int=49, num_of_heads:int=7, dropout:float=0):
        super().__init__()
        assert embedding_dim % num_of_heads == 0,"Incompatible number of heads"
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multi_head_self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_of_heads,
            dropout=dropout
        )
    def forward(self, x):
        x = self.layer_norm1(x)
        attn_output, _ = self.multi_head_self_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output

multihead_selfattn = MultiHeadSelfAttentionBlock(embedding_dim=49, num_of_heads=7)
patch_img_after_attn = multihead_selfattn(patch_and_position_embeds)
# print(patch_img_after_attn.shape)

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim:int=49, expansion_factor:int=4, dropout:float=0.1):
        super().__init__()
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=embedding_dim*expansion_factor),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=embedding_dim*expansion_factor,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm2(x)
        out = self.mlp(x)
        return out
mlpblock = MLPBlock(embedding_dim=49, expansion_factor=4)
attn_after_mlp = mlpblock(patch_img_after_attn)
# print(attn_after_mlp.shape)

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=49,
                 num_of_heads:int=7,
                 expansion_factor:int=4,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1
                 ):
        super().__init__()
        self.multi_head_SA_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_of_heads=num_of_heads,
            dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=mlp_dropout
        )

    def forward(self, x):
        attn_output_with_skip = self.multi_head_SA_block(x) + x
        mlp_output_with_skip = self.mlp_block(attn_output_with_skip) + attn_output_with_skip
        return mlp_output_with_skip


transformer_encoder_block = TransformerEncoderBlock()
# print(summary(transformer_encoder_block, torch.zeros(1, 16, 49), show_input=True, show_hierarchical=True))

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=28,
                 patch_size:int=7,
                 in_channels:int=1,
                 embedding_dim:int=49,
                 num_of_heads:int=7,
                 expansion_factor:int=4,
                 embedding_dropout:float=0.1,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 num_of_transformer_layers:int=6,
                 num_of_classes:int=10):
        super().__init__()
        assert img_size % patch_size == 0, "Incompatible patch size for given image size."
        self.num_of_patches = ((img_size**2) // (patch_size**2))
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.class_embedding = nn.Parameter(data=torch.rand(1, 1, embedding_dim), requires_grad=True)
        self.positional_embedding = nn.Parameter(data=torch.rand(1, self.num_of_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_of_heads=num_of_heads,
            expansion_factor=expansion_factor,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout
        ) for _ in range(num_of_transformer_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_of_classes)
        )

    def forward(self, x):
        # X => [N, C, H, W] => [N, 1, 28, 28]
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        # class_token => [N, C, D] => [N, 1, 49]
        x = self.patch_embedding(x)
        # x => [N, C, P, D] => [N, 1, 16, 49]
        x = torch.cat((class_token, x), dim=1)
        # x + class_token => [N, C, P, D] => [N, 1, 17, 49]
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        # For all batches, computes the classifer output for first sample (Class Token) i.e. Classification Head
        return x

# class ANN(nn.Module):
#     def __init__(self, input_shape: int, hidden_size: int, output_shape: int):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=input_shape, out_features=hidden_size),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden_size, out_features=output_shape),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         return (self.layers(x))
#
#
# torch.manual_seed(42)

# Need to setup model with input parameters
# model_0 = ANN(input_shape=784, hidden_size=20, output_shape=len(new_class_names))
# print(model_0.to("cpu"))
set_seeds()
# random_image_tensor = torch.randn(1, 1, 28, 28)
model_0 = ViT(num_of_classes=len(new_class_names))
# print(vanilla_Vision_Transformer(random_image_tensor))
# print(summary(vanilla_Vision_Transformer, random_image_tensor, show_input=True, show_hierarchical=True))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


EPOCHS = 3
torch.manual_seed(42)

train_start_cpu = timer()
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-------------")
    train_loss = 0
    for batch, (imgs, labels) in enumerate(train_dataloader):
        model_0.train()
        train_preds = model_0(imgs)
        loss = loss_fn(train_preds, labels)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            print(f"Evaluated {batch*len(imgs)} out of {len(train_dataloader.dataset)}")

    train_loss /= len(train_dataloader.dataset)
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for test_imgs, test_labels in test_dataloader:
            test_preds = model_0(test_imgs)
            test_loss = loss_fn(test_preds, test_labels)
            test_acc += accuracy_fn(y_true=test_labels, y_pred=test_preds.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

train_end_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_start_cpu,
                                           end=train_end_cpu,
                                           device="cpu")

