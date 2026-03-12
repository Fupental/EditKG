"""
==========================================================================
parser.py — 命令行参数解析模块
==========================================================================
功能说明：
    定义所有可调的超参数和实验配置。运行 main.py 时可以通过命令行覆盖默认值。
    例如：python main.py --dataset last-fm --lr 0.001 --gpu_id 0

文件角色：
    这是最基础的配置文件，被 main.py 和 evaluate.py 导入使用。
    所有超参数集中在此管理，方便调参和实验对比。
==========================================================================
"""
import argparse


def parse_args():
    """
    解析命令行参数，返回一个包含所有配置的 Namespace 对象。
    使用方式：args = parse_args()，然后通过 args.lr 等方式访问参数。
    """
    parser = argparse.ArgumentParser(description="KRDN")

    # ===== 数据集相关参数 ===== #
    # --dataset: 选择使用哪个数据集，论文实验用了 last-fm, amazon-book, yelp2018 三个
    parser.add_argument("--dataset", nargs="?", default="amazon-book", help="Choose a dataset:[last-fm,alibaba-ifashion,yelp2018,mind-f,amazon-book,MIND]")
    # --data_path: 数据文件存放的根目录，实际路径为 data_path + dataset + '/'
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
 
    # ===== 训练相关参数 ===== #
    # --epoch: 最大训练轮数（实际代码中 main.py 硬编码为100轮，此参数未被使用）
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    # --batch_size: 每批训练样本数量，论文设为4096
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    # --test_batch_size: 测试时每批用户/物品数量
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    # --dim: embedding维度，论文中 d=128（但代码中实际用的是 dim*3=300 因为有三组embedding）
    parser.add_argument('--dim', type=int, default=128, help='embedding size')
    # --l2: L2正则化权重，防止过拟合
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # --lr: 学习率，论文设为 5e-4
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate') #0.0005-->yelp 2018  
    # --gamma: 论文中用于Knowledge Deleter的阈值参数
    parser.add_argument('--gamma', type=float, default=0.5, help='drop threshold')
    # --lr_dc_step: 学习率衰减步数（当前代码中未启用 scheduler）
    parser.add_argument('--lr_dc_step', type=float, default=100, help='drop threshold')
    # --lr_dc: 学习率衰减系数
    parser.add_argument('--lr_dc', type=float, default=0.1, help='drop threshold')
    # --max_iter: GNN消息传递的迭代次数
    parser.add_argument('--max_iter', type=float, default=2, help='iteration times')
    # --inverse_r: 是否考虑知识图谱中的逆向关系（如 <实体, 属于, 物品>）
    parser.add_argument("--inverse_r", type=bool, default=False, help="consider inverse relation or not")
    # --node_dropout: 是否使用节点dropout（在图的稀疏矩阵上随机丢弃边）
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    # --node_dropout_rate: 节点dropout的比例，不同数据集推荐不同值
    parser.add_argument("--node_dropout_rate", type=float, default=1, help="ratio of node dropout") #yelp-->0.75 last-fm-->0.3
    # --mess_dropout: 是否使用消息dropout（在聚合后的embedding上随机置零）
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    # --mess_dropout_rate: 消息dropout比例
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    # --batch_test_flag: 测试时是否分批计算（True=分批，节省显存）
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    # --channel: 模型隐藏层通道数（等同于embedding维度）
    parser.add_argument("--channel", type=int, default=100 , help="hidden channels for model")
    # --cuda: 是否使用GPU加速
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    # --gpu_id: 使用哪块GPU（多卡服务器上通过此参数选择）
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    # --Ks: 评估时的Top-K列表，如 [20, 40] 表示同时计算 Recall@20 和 Recall@40
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Output sizes of every layer')
    # --test_flag: 测试模式，'part'=只对非训练集物品排序（快），'full'=对全部物品排序（慢但更准）
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
 
    # ===== 图神经网络（GNN）相关参数 ===== #
    # --context_hops: GNN的层数，即消息传递的跳数。论文中 L=2
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    # --num_neg_sample: 每个正样本对应的负样本数量
    parser.add_argument('--num_neg_sample', type=int, default=1, help='the number of negative sample')
    # --margin: 对比学习损失的margin参数
    parser.add_argument('--margin', type=float, default=0.2, help='the margin of contrastive_loss')
    # --loss_f: 损失函数类型，可选 contrastive_loss 或 inner_bpr
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, contrastive_loss]")

    # ===== 模型保存参数 ===== #
    # --save: 是否保存训练好的模型参数
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    # --out_dir: 模型参数保存目录
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    # ===== 多模态调试参数 ===== #
    parser.add_argument("--mm_debug", action="store_true", help="打印多模态特征加载和融合的调试信息")
    parser.add_argument("--quick_test", action="store_true", help="快速测试模式：仅跑2轮，验证代码能否跑通")

    return parser.parse_args()