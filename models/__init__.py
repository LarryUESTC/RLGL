from models.Self.E2_SGRL import E2_SGRL
from models.Rein.RLGRL import RLGL
from models.Self.SGRL import SGRL
from models.NodeClas.Semi_GCN import GCN as Semi_GCN
from models.NodeClas.Semi_GCNMIXUP import GCNMIXUP
from models.Rein.RLG import RLG
from models.NodeClas.Sup_GCN import Sup_GCN


method_dict = {
'E2_SGRL':E2_SGRL,
'SEMI_GCN':Semi_GCN,
'SUP_GCN':Sup_GCN,
'GCNMIXUP': GCNMIXUP,
'RLGL':RLGL,
'SGRL':SGRL,
'RLG':RLG,
}

def getmodel(name):
    return method_dict[name.upper()]