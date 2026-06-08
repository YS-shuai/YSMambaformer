from models.Mambaformer.model import Mambaformer
from models.Mambaformer_S.model import Mambaformer_S
from models.Mambaformer_T.model import Mambaformer_T
from models.Mambaformer_ST.model import Mambaformer_ST
from models.Mambaformer_STS.model import Mambaformer_STS
from models.Mambaformer_TSS.model import Mambaformer_TSS
from models.Mambaformer_TST.model import Mambaformer_TST
from models.DynamicNet_multi.model import DynamicNet_multi
from models.RNN.ct_rnn import CT_RNN
MODELS = {
    "Mambaformer": Mambaformer,
    "Mambaformer_S": Mambaformer_S,
    "Mambaformer_T": Mambaformer_T,
    "Mambaformer_ST": Mambaformer_ST,
    "Mambaformer_STS": Mambaformer_STS,
    "Mambaformer_TSS": Mambaformer_TSS,
    "Mambaformer_TST": Mambaformer_TST,
    "DynamicNet_multi": DynamicNet_multi,
    "CT_RNN": CT_RNN,
}
