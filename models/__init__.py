from models.Self.E2_SGRL import E2_SGRL
from models.Rein.RLGRL import RLGL
from models.Self.SGRL import SGRL
from models.Self.SUGRL import SUGRL
from models.Self.GMI import GMI
from models.Self.MVGRL import MVGRL
from models.NodeClas.Semi_GCN import GCN as Semi_GCN
from models.NodeClas.Semi_GCNMIXUP import GCNMIXUP
from models.Rein.RLG import RLG
from models.NodeClas.Sup_GCN import Sup_GCN
from models.VisionGraph.PyramidViG import PyramidViG as IMGCLS_VIG

method_dict = {
'E2_SGRL':E2_SGRL,
'SEMI_GCN':Semi_GCN,
'SUP_GCN':Sup_GCN,
'GCNMIXUP': GCNMIXUP,
'RLGL':RLGL,
'SGRL':SGRL,
'RLG':RLG,
'SUGRL':SUGRL,
'GMI':GMI,
'MVGRL':MVGRL,
}


def getmodel(name):
    return method_dict[name.upper()]
