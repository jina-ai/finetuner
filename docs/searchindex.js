Search.setIndex({docnames:["api/finetuner","api/finetuner.helper","api/finetuner.labeler","api/finetuner.labeler.executor","api/finetuner.tailor","api/finetuner.tailor.base","api/finetuner.tailor.keras","api/finetuner.tailor.paddle","api/finetuner.tailor.pytorch","api/finetuner.toydata","api/finetuner.tuner","api/finetuner.tuner.base","api/finetuner.tuner.dataset","api/finetuner.tuner.dataset.helper","api/finetuner.tuner.keras","api/finetuner.tuner.keras.datasets","api/finetuner.tuner.keras.losses","api/finetuner.tuner.paddle","api/finetuner.tuner.paddle.datasets","api/finetuner.tuner.paddle.losses","api/finetuner.tuner.pytorch","api/finetuner.tuner.pytorch.datasets","api/finetuner.tuner.pytorch.losses","api/finetuner.tuner.summary","api/modules","basics/data-format","basics/fit","basics/glossary","basics/index","components/index","components/labeler","components/overview","components/tailor","components/tuner","design/design-decisions","design/design-philo","design/index","design/overview","get-started/celeba","get-started/covid-qa","get-started/fashion-mnist","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/finetuner.rst","api/finetuner.helper.rst","api/finetuner.labeler.rst","api/finetuner.labeler.executor.rst","api/finetuner.tailor.rst","api/finetuner.tailor.base.rst","api/finetuner.tailor.keras.rst","api/finetuner.tailor.paddle.rst","api/finetuner.tailor.pytorch.rst","api/finetuner.toydata.rst","api/finetuner.tuner.rst","api/finetuner.tuner.base.rst","api/finetuner.tuner.dataset.rst","api/finetuner.tuner.dataset.helper.rst","api/finetuner.tuner.keras.rst","api/finetuner.tuner.keras.datasets.rst","api/finetuner.tuner.keras.losses.rst","api/finetuner.tuner.paddle.rst","api/finetuner.tuner.paddle.datasets.rst","api/finetuner.tuner.paddle.losses.rst","api/finetuner.tuner.pytorch.rst","api/finetuner.tuner.pytorch.datasets.rst","api/finetuner.tuner.pytorch.losses.rst","api/finetuner.tuner.summary.rst","api/modules.rst","basics/data-format.md","basics/fit.md","basics/glossary.md","basics/index.md","components/index.md","components/labeler.md","components/overview.md","components/tailor.md","components/tuner.md","design/design-decisions.md","design/design-philo.md","design/index.md","design/overview.md","get-started/celeba.md","get-started/covid-qa.md","get-started/fashion-mnist.md","index.md"],objects:{"":{finetuner:[0,0,0,"-"]},"finetuner.helper":{AnyDNN:[1,2,1,""],AnyDataLoader:[1,2,1,""],AnyOptimizer:[1,2,1,""],DocumentArrayLike:[1,2,1,""],DocumentSequence:[1,2,1,""],LayerInfoType:[1,2,1,""],get_framework:[1,1,1,""],is_seq_int:[1,1,1,""]},"finetuner.labeler":{executor:[3,0,0,"-"],fit:[2,1,1,""]},"finetuner.labeler.executor":{DataIterator:[3,3,1,""],FTExecutor:[3,3,1,""]},"finetuner.labeler.executor.DataIterator":{add_fit_data:[3,4,1,""],requests:[3,5,1,""],store_data:[3,4,1,""],take_batch:[3,4,1,""]},"finetuner.labeler.executor.FTExecutor":{embed:[3,4,1,""],fit:[3,4,1,""],get_embed_model:[3,4,1,""],requests:[3,5,1,""],save:[3,4,1,""]},"finetuner.tailor":{base:[5,0,0,"-"],display:[4,1,1,""],keras:[6,0,0,"-"],paddle:[7,0,0,"-"],pytorch:[8,0,0,"-"],to_embedding_model:[4,1,1,""]},"finetuner.tailor.base":{BaseTailor:[5,3,1,""]},"finetuner.tailor.base.BaseTailor":{display:[5,4,1,""],embedding_layers:[5,6,1,""],summary:[5,4,1,""],to_embedding_model:[5,4,1,""]},"finetuner.tailor.keras":{KerasTailor:[6,3,1,""]},"finetuner.tailor.keras.KerasTailor":{summary:[6,4,1,""],to_embedding_model:[6,4,1,""]},"finetuner.tailor.paddle":{PaddleTailor:[7,3,1,""]},"finetuner.tailor.paddle.PaddleTailor":{summary:[7,4,1,""],to_embedding_model:[7,4,1,""]},"finetuner.tailor.pytorch":{PytorchTailor:[8,3,1,""]},"finetuner.tailor.pytorch.PytorchTailor":{summary:[8,4,1,""],to_embedding_model:[8,4,1,""]},"finetuner.toydata":{generate_fashion_match:[9,1,1,""],generate_qa_match:[9,1,1,""]},"finetuner.tuner":{base:[11,0,0,"-"],dataset:[12,0,0,"-"],fit:[10,1,1,""],keras:[14,0,0,"-"],paddle:[17,0,0,"-"],pytorch:[20,0,0,"-"],save:[10,1,1,""],summary:[23,0,0,"-"]},"finetuner.tuner.base":{BaseDataset:[11,3,1,""],BaseLoss:[11,3,1,""],BaseTuner:[11,3,1,""]},"finetuner.tuner.base.BaseLoss":{arity:[11,5,1,""]},"finetuner.tuner.base.BaseTuner":{arity:[11,6,1,""],embed_model:[11,6,1,""],fit:[11,4,1,""],save:[11,4,1,""]},"finetuner.tuner.dataset":{SiameseMixin:[12,3,1,""],TripletMixin:[12,3,1,""],helper:[13,0,0,"-"]},"finetuner.tuner.dataset.helper":{get_dataset:[13,1,1,""]},"finetuner.tuner.keras":{KerasTuner:[14,3,1,""],datasets:[15,0,0,"-"],losses:[16,0,0,"-"]},"finetuner.tuner.keras.KerasTuner":{fit:[14,4,1,""],save:[14,4,1,""]},"finetuner.tuner.keras.datasets":{SiameseDataset:[15,3,1,""],TripletDataset:[15,3,1,""]},"finetuner.tuner.keras.losses":{CosineSiameseLoss:[16,3,1,""],CosineTripletLoss:[16,3,1,""],EuclideanSiameseLoss:[16,3,1,""],EuclideanTripletLoss:[16,3,1,""]},"finetuner.tuner.keras.losses.CosineSiameseLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.CosineTripletLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.EuclideanSiameseLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.keras.losses.EuclideanTripletLoss":{arity:[16,5,1,""],call:[16,4,1,""]},"finetuner.tuner.paddle":{PaddleTuner:[17,3,1,""],datasets:[18,0,0,"-"],losses:[19,0,0,"-"]},"finetuner.tuner.paddle.PaddleTuner":{fit:[17,4,1,""],save:[17,4,1,""]},"finetuner.tuner.paddle.datasets":{SiameseDataset:[18,3,1,""],TripletDataset:[18,3,1,""]},"finetuner.tuner.paddle.losses":{CosineSiameseLoss:[19,3,1,""],CosineTripletLoss:[19,3,1,""],EuclideanSiameseLoss:[19,3,1,""],EuclideanTripletLoss:[19,3,1,""]},"finetuner.tuner.paddle.losses.CosineSiameseLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.CosineTripletLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanSiameseLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.paddle.losses.EuclideanTripletLoss":{arity:[19,5,1,""],forward:[19,4,1,""]},"finetuner.tuner.pytorch":{PytorchTuner:[20,3,1,""],datasets:[21,0,0,"-"],losses:[22,0,0,"-"]},"finetuner.tuner.pytorch.PytorchTuner":{fit:[20,4,1,""],save:[20,4,1,""]},"finetuner.tuner.pytorch.datasets":{SiameseDataset:[21,3,1,""],TripletDataset:[21,3,1,""]},"finetuner.tuner.pytorch.losses":{CosineSiameseLoss:[22,3,1,""],CosineTripletLoss:[22,3,1,""],EuclideanSiameseLoss:[22,3,1,""],EuclideanTripletLoss:[22,3,1,""]},"finetuner.tuner.pytorch.losses.CosineSiameseLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.CosineTripletLoss":{forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanSiameseLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.pytorch.losses.EuclideanTripletLoss":{arity:[22,5,1,""],forward:[22,4,1,""]},"finetuner.tuner.summary":{NumericType:[23,2,1,""],ScalarSummary:[23,3,1,""],SummaryCollection:[23,3,1,""]},"finetuner.tuner.summary.ScalarSummary":{floats:[23,4,1,""]},"finetuner.tuner.summary.SummaryCollection":{dict:[23,4,1,""],save:[23,4,1,""]},finetuner:{fit:[0,1,1,""],helper:[1,0,0,"-"],labeler:[2,0,0,"-"],tailor:[4,0,0,"-"],toydata:[9,0,0,"-"],tuner:[10,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","data","Python data"],"3":["py","class","Python class"],"4":["py","method","Python method"],"5":["py","attribute","Python attribute"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:function","2":"py:data","3":"py:class","4":"py:method","5":"py:attribute","6":"py:property"},terms:{"0":[0,9,10,14,16,17,19,20,22,25,30,32,33,38,40,41],"00":[30,38,40],"000":25,"00010502179":25,"001":[0,10,14,17,20],"002585097":25,"011804931":25,"028714137":25,"03":38,"06":38,"08":[10,14,17,20],"0e7ec5aa":25,"0e7ec7c6":25,"0e7ecd52":25,"0e7ece7":25,"1":[9,16,19,22,26,30,32,33,34,38,39,40,41],"10":[0,10,11,14,17,20,25,33],"100":[9,26,30,32,38,39,41],"1000":[32,38],"100480":32,"102764544":32,"109":[30,38,40],"11":38,"112":32,"1180160":32,"11ec":[25,39,40],"128":[32,33,34,40,41],"12900":32,"132":[30,38,40],"135":[30,38,40],"14":32,"141":40,"147584":32,"16781312":32,"172":[30,38,40],"1792":32,"18":[30,38,40],"180":38,"19":40,"1bab":40,"1bae":39,"1e":[10,14,17,20],"1e008a366d49":[25,39,40],"1f9f":25,"1faa":25,"2":[9,11,16,19,22,25,26,32,33,39,40,41],"20":38,"2021":34,"224":[32,38],"22900":40,"231":[30,38,40],"2359808":32,"25088":32,"2508900":32,"256":[0,10,11,14,17,20,32],"28":[32,33,40,41],"28x28":25,"295168":32,"29672":30,"3":[9,11,16,19,22,25,26,32,34,38,40,41],"31":40,"32":[32,33,39,40,41],"320000":32,"33":40,"3300":32,"36928":32,"3d":35,"3gb":38,"4":[25,26,38,41],"4096":32,"409700":32,"4097000":32,"4128":32,"481":[9,25],"49":40,"5":[25,33,41],"5000":[32,33,39],"512":32,"52621":40,"53":38,"56":[9,32,40],"5716974480":40,"5794172560":39,"590080":32,"6":[25,38],"60":25,"60000":9,"61130":30,"61622":38,"64":[32,33,39],"65":40,"66048":32,"6620":38,"66560":32,"67":40,"67432a92":25,"67432cd6":25,"685":[30,38],"7":[32,40,41],"70":35,"73856":32,"75":38,"784":32,"784x128x32":33,"9":[10,14,17,20,40,41],"94":[30,38,40],"99":[10,14,17,20,35],"999":[10,14,17,20],"9a49":25,"abstract":[3,5,11],"case":[35,41],"class":[3,5,6,7,8,11,12,14,15,16,17,18,19,20,21,22,23,25,32,33,39,40],"default":[33,34],"do":[25,26,32,34,35,37,38,41],"final":[16,19,22,32,35,38,39,40],"float":[0,10,14,17,20,23,25],"function":[1,3,11,14,17,20,25,32,33],"import":[1,25,26,30,32,33,34,35,37,38,39,40,41],"int":[0,5,6,7,8,9,10,11,14,17,20,23,32],"new":[30,32,33,34,35,41],"public":[30,34,35,37,38,40,41],"return":[0,1,2,4,5,6,7,8,9,10,11,14,17,19,20,22,23,32,33,39],"switch":[33,41],"true":[0,1,9,25,26,30,32,33,34,38,39,40,41],"try":34,"while":30,A:[19,22,27,30],And:35,As:34,At:34,But:[25,26,32,34,41],By:[33,34,37],For:[9,11,25,27,30,32,34,38],If:[9,25,30,35,38,39],In:[25,30,32,33,34,35,38,39,40,41],It:[1,5,6,7,8,9,25,30,32,33,34,35,37,39,41],Its:33,No:[26,41],On:[34,38],One:9,Or:35,That:[34,35],The:[1,5,6,7,8,9,10,14,16,17,19,20,22,23,25,30,31,33,34,35,38,39,40,41],Then:[25,35],There:[34,35],To:[5,6,7,8,25,32,33,34,37],With:35,_:[32,33,39],__init__:7,__module__:1,_i:33,_j:33,_n:33,_p:33,a207:40,a46a:25,a5dd3158:25,a5dd3784:25,a5dd3b94:25,a5dd3d74:25,aaaaaaaa:25,aaaaaaaaaaaaaa:40,aaaaaaaaaaaaaaaa:39,aaaaaaaaekaaaaaaaaawqaaaaaaaabpa:25,abc:[5,11],abl:[30,38],about:[26,33,34,35,37,41],abov:[30,38],ac8a:25,accept:[25,33,38,39,40],access:[30,38,40],accompani:41,accord:[33,34],accur:32,accuraci:[35,40],achiev:32,across:34,action:32,activ:[30,31,32,33,34,37,38,40,41],actual:[1,34],ad:[32,35],adam:[0,10,14,17,20],adaptiveavgpool2d_32:32,add:[25,32],add_fit_data:3,addit:[3,32],address:[30,38,40],adjac:25,adjust:30,adopt:34,advanc:25,affect:[30,34,35],after:[5,6,7,8,25,30,32,33,34,35,37,38,39,40],afterward:[30,32],ag:35,again:[32,35],agnost:[34,41],ai:[34,35,37,41],aim:[34,37],algorithm:30,alia:[1,23],all:[5,6,7,8,9,16,19,22,23,26,30,32,33,34,38,41],allow:[11,25,34,35,37,41],alreadi:[26,30,34,41],also:[16,19,22,30,32,33,34,35,39],although:33,alwai:[3,25],among:35,an:[1,5,6,7,8,10,26,27,30,31,32,33,34,35,37,41],anchor:[16,19,22],ani:[1,5,6,7,8,26,27,30,31,32,34,35,37,39,40,41],annot:34,answer:[25,26,30,35,41],anydataload:1,anydnn:[0,1,4,5,6,7,8,10,11,14,17,20,32],anymor:35,anyon:[35,37],anyoptim:1,anyth:25,apach:41,apart:32,api:[25,32,35,37,41],app:[35,37,41],append:25,appli:35,applic:[32,35,38,41],aqaaaaaaaaacaaaaaaaaaamaaaaaaaaa:25,ar:[3,5,6,7,8,10,11,14,17,20,25,26,30,32,33,34,35,38,39,40,41],ara:3,arbitari:34,arbitrari:34,architectur:[5,6,7,8,32,34],arg:[5,10,11,14,16,17,19,20,22],argu:37,argument:[3,10,14,17,20,30,38],ariti:[11,13,16,19,22],around:35,arrai:[1,25,38,39,40],articl:35,arxiv:35,ask:[27,30],async:30,attract:35,auf:25,auto:3,avail:[5,6,7,8,11,14,17,20,30,38,40],averag:[16,19,22],avoid:3,awar:35,axi:9,b32d:39,b9557788:40,b:[9,25,27,34],baaaaaaaaaafaaaaaaaaaayaaaaaaaaa:25,back:[34,41],backend:[30,37,38,40,41],bad:[25,30],baidu:34,bar:[30,39],base64:40,base:[0,1,3,4,6,7,8,10,12,14,15,16,17,18,19,20,21,22,23,30,38,39,40],basedataset:[11,15,18,21],baseexecutor:3,baseloss:[11,16,19,22],basetailor:[5,6,7,8],basetun:[11,14,17,20],batch:[5,6,7,8,10,14,16,17,19,20,22,27,34],batch_first:[32,33,39],batch_siz:[0,10,11,14,17,20],beautifulli:34,becam:35,becaus:[25,30],becom:35,been:[30,32,39],befor:[35,38,39,40],behav:37,behavior:34,behind:34,being:[16,19,22,35],believ:35,beliv:35,belong:[16,19,22],below:[25,26,30,33,34,35,37,41],bert:35,besid:30,best:[34,38],beta_1:[10,14,17,20],beta_2:[10,14,17,20],better:[11,16,19,22,27,30,31,34,35,37,38,39,40,41],between:[16,19,22,25,33,41],bewar:38,bidirect:[32,39],big:[33,34],bigger:9,bit:35,blank:35,blob:[25,39,40],block1_conv1:32,block1_conv2:32,block1_pool:32,block2_conv1:32,block2_conv2:32,block2_pool:32,block3_conv1:32,block3_conv2:32,block3_conv3:32,block3_pool:32,block4_conv1:32,block4_conv2:32,block4_conv3:32,block4_pool:32,block5_conv1:32,block5_conv2:32,block5_conv3:32,block5_pool:32,block:35,blog:34,bonjour:25,bool:[0,1,5,6,7,8,9,32],both:[16,19,22],bottleneck:[32,34],brows:35,browser:[30,38],buffer:[25,39,40],build:[25,32,33,34,35,41],built:[25,33,35],butteri:41,button:[30,35],c:[5,6,7,8,9],calendar:41,call:[11,16,25,32,33,34],callabl:[1,10,14,17,20],cambridg:35,can:[5,6,7,8,9,25,26,30,31,32,33,34,35,37,38,39,40,41],cannot:25,card:30,carefulli:34,center:[10,14,17,20],certain:32,chain:[34,35],chanc:35,chang:[11,30,35],channel:[9,25,41],channel_axi:9,chapter:[25,35,37],chat:41,chatbot:39,check:[5,6,7,8],checkout:[39,40],china:34,choic:30,choos:38,chop:32,chosen:34,clarifi:34,classic:35,classif:[27,34],clear_labels_on_start:[0,2,3],click:[30,35],cloud:35,co:33,code:[30,34,38,39],codebas:25,colleagu:34,collect:[23,30,34,35,37,38],color:35,color_axi:38,column:25,combin:[25,31],come:[25,30,34],comfort:[37,38],common:[34,35],commun:[34,41],compani:35,compat:34,complet:30,complex:23,compli:34,complic:35,compon:[27,30,31,32,33,34,37],compos:[31,37],comput:[16,19,22,25,30],concat:32,concept:34,conduct:[30,31,34,37],config:[3,11,30],consecut:11,consid:[25,30,34],consider:30,consist:[32,34,37],consol:30,construct:[25,33],contain:[16,19,22,25,27,30,38],content:[24,25,27,30,34,35,37,39],context:[25,41],continu:35,contrari:34,contrast:[16,19,22],conv2d_11:32,conv2d_13:32,conv2d_15:32,conv2d_18:32,conv2d_1:32,conv2d_20:32,conv2d_22:32,conv2d_25:32,conv2d_27:32,conv2d_29:32,conv2d_3:32,conv2d_6:32,conv2d_8:32,converg:[38,39,40],convert:[5,6,7,8,25,26,27,31,32,34,38,39,40,41],convert_image_datauri_to_blob:38,copi:[38,40],core:[35,37],correct:[30,32],correctli:[30,35],correspond:11,cos_sim:[16,19,22],cosim:[16,19,22],cosin:[3,16,19,22,25],cosinesiameseloss:[0,2,3,10,11,14,16,17,19,20,22,33],cosinetripletloss:[11,14,16,17,19,20,22,33],cost:[35,41],could:34,covid19:39,covidqa:39,cpp:[38,40],cpu:[0,10,14,17,20,38],creat:[11,14,17,20,23,35],critic:35,csrc:[38,40],csv:25,cuda:[10,14,17,20],current:[11,14,17,20,35],cut:34,d8aaaaaaaaeqaaaaaaaaaxa:25,d:[16,19,22,25,27,34,38],da1:25,da2:25,dai:35,dam_path:3,danger:30,data:[1,5,6,7,8,9,10,11,14,17,20,21,23,26,27,30,31,33,34,37,41],data_gen:38,dataiter:3,dataset:[0,10,14,17,20,25,38,39,40,41],dc315d50:39,debug:[25,30],decis:[35,37],decompress:35,decoupl:35,deep:[1,25,31,34,35,37,41],def:[32,33,38,39],defacto:35,defailt:[10,14,17,20],defin:[5,6,7,8,32],definit:34,deliv:[34,35,41],deliveri:[34,37],demo:[25,39,40],denot:33,dens:[5,25,32,33,39,40,41],dense_1:32,dense_2:32,depend:[30,34],design:[30,35,37,41],desir:[16,19,22,32],despit:34,detect:[27,34],determin:[30,34],develop:[33,34,35,37],devic:[0,10,14,17,20],dict:[0,1,3,5,6,7,8,10,14,17,20,23],dictionari:23,differ:[11,16,19,22,25,30,31,32,34],dim:[32,33,39,40,41],dimens:[27,32],dimension:[5,6,7,8,32,39,41],direct:[32,33,39],directli:[25,30,32,34],discuss:41,displai:[4,5],dist:[16,19,22],dist_neg:[16,19,22],dist_po:[16,19,22],distanc:[16,19,22],dive:41,divers:30,diversifi:34,divid:30,dnn:[1,5,6,7,8,27,34,35],dnn_model:1,doc:[3,30,38,40],doctor:39,document:[1,9,25,27,30,33,34,38,39,40],documentarrai:[1,25,27,34,38,39,40],documentarraylik:[0,1],documentarraymemap:25,documentarraymemmap:[1,25,34,38,39,40],documentsequ:[1,10,14,17,20],doe:[9,26,34,38,40,41],domain:[35,41],don:[26,35,41],done:[30,34,35,38,40],download:[35,38],downstream:34,dropout_35:32,dropout_38:32,dtype:[25,39,40],dure:3,e:[9,23,25,34],each:[9,10,14,17,20,25,27,30,33,34,38],earlier:35,easi:[26,35,41],easier:30,easili:37,ecosystem:[34,35,37,41],edg:35,effect:35,effort:34,either:[11,14,17,20,25,32,33],ell_:33,emb:3,embed:[1,5,6,7,8,10,11,14,16,17,19,20,22,25,26,27,30,31,32,33,34,35,37,38,41],embed_model:[2,10,11,14,17,20,26,30,32,33,34,39,40,41],embedding_1:32,embedding_dim:[32,33,39],embedding_lay:[5,6,7,8],enabl:38,end:[10,14,17,20,32,34],endli:34,endpoint:3,engin:[35,41],enjoi:37,enlarg:34,enough:35,enpow:1,ensur:[16,19,22],entrypoint:34,enviro:35,epoch:[0,10,11,14,17,20,30,33],epsilon:[10,14,17,20],equal:[16,19,22,34],est:34,establish:35,estim:[30,38],euclidean:[16,19,22],euclideansiameseloss:[11,14,16,17,19,20,22,33],euclideantripletloss:[11,14,16,17,19,20,22,33],eval_data:[0,10,11,14,17,20,25,33],evalu:[10,14,17,20,25],event:41,everi:[25,30,35,39,40,41],everyth:[30,35],exactli:34,exampl:[9,11,27,30,34,38,39,40,41],except:32,exchang:34,executor0:30,executor1:30,executor:[0,2,30,35,38,40],exhaust:33,exist:[32,33,34,35],expect:[3,9,35],experi:[30,34,38,41],explain:35,expos:[34,35],exposur:34,express:25,extend:25,exteremli:35,extra:3,extrem:[26,34,35,41],f4:40,f8:25,f:20,factor:9,fail:3,fals:[0,2,3,4,5,6,7,8,9,10,14,17,20,25,32],familiar:35,far:30,fashion:[9,41],faster:30,fc1:32,fc2:32,featur:[35,41],feed:[3,25,33,39,40,41],feedback:[25,35,39,40],feel:[34,37,38],fetch:35,few:[30,34],fewer:35,field:[3,5,6,7,8,30],file:[10,23,30,35],filepath:[14,23],fill:[3,25,27,34,35,39],find:[26,30,32,41],fine:[26,32,41],finetun:[25,26,27,30,31,32,33,34,37],first:[16,19,22,32,33,34,35,38,39,40],fit:[0,2,3,10,11,14,17,20,25,27,32,34,37,38,39,40,41],fix:[9,25,34],flatten:[32,33,40,41],flatten_1:32,flatten_input:32,flexibl:34,float32:[0,4,5,6,7,8,32],flow:[30,34,38,40],fly:[25,26,34,41],folder:10,follow:[25,26,30,32,33,34,35,38,39,40,41],form:[33,34],format:[33,34,38,39,40],forward:[19,22,32,33,39],forwardref:0,found:[33,35],four:25,frac:33,framework:[1,10,32,34,38,39,40,41],freeli:33,freez:[0,4,5,6,7,8,32,38,41],freeze_lay:34,from:[3,5,6,7,8,11,14,17,20,23,25,30,32,33,34,35,37,38,39,40,41],from_fil:38,frontend:[25,30,41],frozen:32,ft:[34,37],ftexecutor:3,full:[35,38],fulli:34,fundament:32,funnel:34,further:30,g:[9,23],gap:34,gatewai:30,gener:[5,6,7,8,9,25,27,30,32,33,34,35,37,38,39,40],general_model:[26,30,34,41],generate_fashion_match:[9,25,33,40,41],generate_qa_match:[9,25,33,39],get:[5,9,11,30,32,34,35,38,39,40,41],get_dataset:13,get_embed_model:3,get_framework:1,give:[1,9,32,34,37],given:[27,30,32,33,34,38,40],global:35,go:[39,40],good:[25,34,35],goodby:25,googl:35,got:[26,41],gpu:[10,14,17,20,38],grai:9,grammar:35,graph:25,grayscal:[9,25],grid:30,groundtruth:25,h236cf4:25,h:9,ha:[25,30,32,34,37],had:35,hallo:25,hand:[35,41],handl:[30,34],hanxiao:[30,38,40],happen:34,have:[9,26,30,32,34,35,38,39,41],heavi:35,hello:[25,39,40],help:[26,38,41],helper:[0,10,11,12,24,30,32,38],henc:[25,34],here:[16,19,22,25,32,33,34,35,37,38],hf:35,high:[25,32,34,35,37],highlight:35,hire:41,hit:30,hopefulli:[38,39,40],how:[25,30,32,34,35,37,38,41],howev:[25,33,35],http:[30,38,40],httpruntim:30,hub:35,huggingfac:[32,34,35],human:[27,30,34,35,41],i8:[25,39],i:[25,30,33,34,35,37,38,40],ical:41,id:[25,39,40],idea:[1,35,41],ident:[5,6,7,8,41],identityceleba:38,ignor:[30,38],illustr:25,imag:[9,25,27,30,33,34,38],imaga:9,imagenet:38,img_align_celeba:38,implement:[1,30,32,33,34,38],impli:37,improv:[25,35,38,39,40,41],in_featur:[32,33,39,40,41],includ:[5,6,7,8,23,25],include_identity_lay:[5,6,7,8],incorp:35,independ:[34,35],index:41,indic:30,influr:35,info:[5,6,7,8],inform:[1,25,32],initi:[23,33,34],inject:3,input:[1,5,6,7,8,11,14,15,16,17,18,19,20,21,22,25,27,30,32,33,34],input_dim:[32,33,39],input_dtyp:[0,4,5,6,7,8,32],input_s:[0,4,5,6,7,8,32,38],input_shap:[32,33,40,41],insid:[34,35,37],inspect:38,inspir:[34,39,40],instal:41,instanc:[11,14,17,20,25,30],int64:32,integ:[1,5,6,7,8,9],integr:41,interact:[0,25,26,31,34,37,41],interest:34,interfac:[34,35],intern:[38,40],introduc:25,intuit:41,invert:30,invok:34,io:18,irrelev:34,is_seq_int:1,is_sim:[16,19,22],is_testset:[9,33],isspeci:30,item:25,iter:[1,32],its:[25,32,33,34,35,38,40],ivborw0k:40,j:33,jina:[1,3,25,30,35,37,38,39,40,41],job:34,jpg:38,json:23,just:[30,35],k:[30,35,38,39,40],keep:[30,35],kei:[1,14,17,20,23,30,33],kera:[0,1,4,10,32,33,34,37,38,39,40,41],kerastailor:6,kerastun:14,keyboard:[30,38,39,40],keyword:[3,10,14,17,20],knowledg:35,known:[16,19,22],kwarg:[2,3,4,5,10,11,14,16,17,19,20,22,33,34],label:[0,9,11,24,26,27,31,33,37,41],labeled_dam_path:3,labeled_data:[26,30,41],labler:38,lambda:33,landscap:[34,35,37],languag:25,larger:30,last:[5,6,7,8,32,34,37,38,41],lastcel:[32,33,39],lastcell_3:32,later:[34,35,37],latest:[34,41],layer:[1,5,6,7,8,16,32,33,34,38,39,40,41],layer_nam:[0,4,5,6,7,8,32],layerinfotyp:1,learn:[10,14,17,20,25,26,30,31,34,37,41],learning_r:[0,10,14,17,20],least:25,left:[30,33,39],length:[9,25],less:[34,35],let:[32,33,34,35,38,39,40,41],level:[25,32,34,35,37,41],leverag:[34,35,38],licens:41,like:[25,27,30],linear:[32,33,38,39,40,41],linear_2:32,linear_33:32,linear_34:32,linear_36:32,linear_39:32,linear_4:32,linear_5:32,liner:41,linux:41,list:[1,5,6,7,8,16,19,22,23,30,32],liter:35,live:41,load:[30,32,35],loader:1,local:[30,38,40],localhost:[30,38,40],logic:25,look:[30,32,34,35,37,39,40],loop:41,loss:[0,2,3,10,11,14,17,20,23,38,40],lstm_2:32,luckili:[26,41],m1:25,m2:25,m3:25,m:25,machin:[35,38],maco:41,made:25,mai:[30,32,34,35,37,38,40],main:30,maintain:34,major:34,make:[33,34,35,38,40,41],manag:34,mandatori:34,mani:[30,35],manner:37,manual:[25,30,32],map:3,margin:[16,19,22],match:[9,27,30,33],mathbf:33,max:[16,19,22,33],max_seq_len:9,maxim:34,maximum:[9,30],maxpool2d_10:32,maxpool2d_17:32,maxpool2d_24:32,maxpool2d_31:32,maxpool2d_5:32,me:[34,35],mean:[25,30,34,38,39,40],meaning:30,meant:[34,35,37],meanwhil:[26,41],meet:41,member:41,memmap:1,mention:35,mesh:35,meta:3,method:[10,14],metric:[3,23],micro:[32,34],microsoft:35,mile:[34,37,41],mime_typ:25,minimum:33,minor:34,minut:35,mission:35,mix:30,mlp:[39,41],mnist:[9,40,41],model:[0,1,4,5,6,7,8,10,11,14,17,20,25,26,27,30,31,33,34,35,37,41],model_path:10,modul:[24,32,33,39,41],momentum:[10,14,17,20],mond:25,monet:35,month:41,more:[25,26,30,34,41],most:[30,34,35,37],mostli:34,motiv:35,mous:[35,38,39,40],move:[10,14,17,20],movi:34,multi:30,multipl:11,multipli:35,must:[9,33,34,37],my:[34,35,37],myself:35,n:[16,19,22,33,35],name:[1,5,6,7,8,11,14,17,20,23,32],nativ:[23,35],natuar:35,nb_param:32,ndarrai:[9,25,27,34],nearest:[25,30],need:[7,14,17,20,25,26,30,32,33,34,35,38,41],neg:[9,16,19,22,25,30,33],neg_valu:9,neighbour:[25,30],nesterov:[10,14,17,20],network:[1,11,16,19,22,25,30,31,34,37,38,40,41],neural:[1,25,31,34,35,37,41],newli:30,next:[3,27,34,35],nich:35,nn:[19,22,32,33,39,40,41],non:[32,38,40],none:[0,2,3,4,5,6,7,8,9,10,11,14,17,20,23,32],note:[5,6,7,8,11,25,30,38,39,40],noth:[34,35],nov:34,now:[25,26,32,35,38,39,40,41],nowher:35,np:25,num_embed:[32,33,39],num_neg:[9,33],num_po:[9,33],num_tot:9,number:[9,10,14,16,17,19,20,22,23,25,30],numer:23,numerictyp:23,numpi:[23,25,38,40],object:[1,11,12,14,16,17,19,20,22,23,25,27,34,39,40],observ:[30,32,34],obviou:35,off:32,often:[25,32,34,35],okayish:35,onc:30,one:[11,25,33,34,35,37,41],onli:[25,30,33,34,35,38],open:[30,38],opensourc:41,oper:[30,32,34],optim:[0,1,10,14,16,17,19,20,22,33],optimizer_kwarg:[0,10,14,17,20],option:[0,5,6,7,8,9,10,11,14,17,20,23,32,34],organ:25,origin:[5,6,7,8,25,32,38],other:[25,33,34,35,41],otheriws:[16,19,22],otherwis:25,our:[34,35,38,39,40,41],out:[26,32,33,35,39,41],out_featur:[32,33,39,40,41],output:[5,6,7,8,26,27,30,32,34,41],output_dim:[0,4,5,6,7,8,26,30,32,33,34,38,39,41],output_shape_displai:32,over:[16,19,22,32],own:[25,32],p:33,packag:24,pad:39,paddl:[0,1,4,10,32,33,34,37,38,39,40,41],paddlepaddl:41,paddletailor:[5,6,7,8],paddletun:17,page:35,pair:[16,19,22,33],paper:35,paragraph:[34,35,37],param:3,paramet:[1,3,5,6,7,8,9,10,11,14,16,17,19,20,22,23,32,33],parent:25,part:[5,6,7,8,30,35,37],partial:34,particular:[25,34,35],pass:[3,5,10,14,17,20],path:[10,14,17,20,30],peopl:[34,35],per:[9,25],perceptron:32,perfect:[26,41],perform:[30,32,34,35,41],philosophi:37,pictur:35,pip:41,pipelin:[34,35,39,40],place:11,plain:25,pleas:[34,35,37],plu:35,png:40,point:35,pool:30,popular:[34,35],port_expos:[0,2],pos_valu:9,pose:35,posit:[9,16,19,22,25,30,33,34,37],positv:30,post:[34,35],potenti:[5,6,7,8,35],power:41,pre:38,preachitectur:34,precis:[39,40],predict:[27,32,33,34],prepar:25,preserv:32,press:34,pretrain:[33,34,35,41],previou:[32,34],primit:25,print:25,prioriti:35,privat:[30,38,40],probabl:35,problem:[32,33,35],procedur:[30,35,38,39,40],process:25,produc:[11,14,17,20,38,39,40],product:[30,35,41],program:[38,40],project:[31,34,35,37],promis:41,promot:34,properti:[5,11],propos:30,protect:[38,40],protocol:[30,38,40],provid:[25,26,32,34,35,37,41],prune:41,publish:[34,35,37],purpos:[25,30,35],py:[30,38,40],python:[23,33,41],pytorch:[0,4,9,10,32,33,34,35,37,38,39,40,41],pytorchtailor:[5,6,7,8],pytorchtun:20,qa:[9,39],qualiti:[34,39,40],quantiti:34,queri:41,question:[25,26,27,35,41],quickli:[26,35,41],r:34,rais:[30,38,40],randomli:25,rate:[10,14,17,20],ratio:30,re:41,reach:[35,37],read:[34,37],readi:[30,35,38,40],real:9,rearrang:35,reason:35,recommend:[30,39,40],record:23,redoc:[30,38,40],reduc:[34,41],reduct:41,refer:33,reflect:25,regress:[27,34],reject:[38,39,40],relat:[25,30,33],releas:34,relev:[25,30],reli:25,relu:[32,33,40,41],relu_12:32,relu_14:32,relu_16:32,relu_19:32,relu_21:32,relu_23:32,relu_26:32,relu_28:32,relu_2:32,relu_30:32,relu_34:32,relu_37:32,relu_3:32,relu_4:32,relu_7:32,relu_9:32,remain:[30,32],rememb:[39,40],remov:[5,6,7,8,32],render:[30,39],repeat:35,replac:32,repres:[16,19,22,25,30,33],represent:34,request:3,requir:[5,6,7,8,32,34,38],rescal:9,research:35,reset:33,resnet50:38,respect:[33,38],respons:[34,35],rest:[38,40],restrict:27,result:[30,34,35,38,39,40],retrain:35,revis:[34,35,37],rgb:9,rho:[10,14,17,20],rich:[34,41],right:[30,33],rmsprop:[10,14,17,20],round:[38,39,40],row:25,run:38,runtim:[3,30],runtime_arg:3,runtime_backend:[0,2],runtimebackend:30,runtimeerror:30,s:[10,25,27,30,32,34,35,38,39,40,41],sai:32,said:34,same:[16,19,22,25,27,35,37,39,40],sampl:[25,30,34],save:[3,10,11,14,17,20,23,30,35],save_path:33,scalar:23,scalarsummari:23,scale:9,scenario:31,schedul:34,score:25,scratch:[32,34,35,37],script:38,search:[25,34,35,37,39,41],second:[16,19,22,30,38,41],section:[30,34,35,37],see:[5,6,7,8,30,32,35,38,40,41],seen:34,select:[30,32,34,39],self:[32,33,39],semant:39,sens:35,sentenc:25,sequenc:[1,5,6,7,8,9,27,34],sequenti:[32,33,39,40,41],serv:34,session:30,set:[5,6,7,8,9,32,35,39,40,41],set_wakeup_fd:30,setup:30,sever:[38,39,40],sgd:[10,14,17,20],shall:[34,35],shape:[5,6,7,8,25,27,39,40],share:[25,34,35,37],shortcut:30,shot:[25,30],should:[9,16,19,22,25,30,34,35,37,38],show:[30,34],siames:[11,16,19,22,33,41],siamesedataset:[15,18,21],siamesemixin:[12,15,18,21],side:30,signal:30,similar:[16,19,22,34,38,39,40],simpl:[34,35],simpli:[25,26,32,34,41],simul:9,singl:[25,37],size:[5,6,7,8,10,14,17,20,27,30,34,38],skip_identity_lay:[6,7,8],slack:41,slide:34,slower:30,smaller:[30,38],smooth:41,so:[25,30,34,35,38,39,40],soldier:34,solid:1,solut:[35,41],solv:[32,33,35,37],some:[16,19,22,25,26,30,32,34,35,37,41],sometim:34,sourc:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,34],space:30,spawn:38,specif:[10,30,32,34,41],specifi:[30,33,38],speed:34,spinner:30,spy:34,stabil:[16,19,22],start:[25,30,38],stat:30,statist:30,step:[32,33,35,38,40],stick:37,still:[30,35],store:[23,25,30,39],store_data:3,str:[0,1,5,6,7,8,10,11,14,17,20,23,32],stream:41,strong:[34,35],stronli:[39,40],structur:34,stuck:35,submiss:30,submit:30,submodul:24,subpackag:24,subscrib:41,success:30,summar:[33,34],summari:[0,5,6,7,8,10,25,32],summarycollect:[0,10,11,14,17,20,23],suppli:[34,35],support:[10,14,17,20,37,38,40],suppos:30,supposedli:[38,40],suppress:[38,40],sure:[33,38,41],swagger:[30,38,40],synergi:35,syntax:35,synthet:[9,25],system:30,t:[26,35,38,41],tabl:[5,32],tag:[25,27,30,39,40],tailor:[0,1,24,26,27,31,37,38,41],take:[34,35,38],take_batch:3,taken:25,talk:[34,35,41],target:[16,19,22],task:[25,34,35,39,40,41],tell:[1,30,39],tensor:[5,6,7,8,16,19,22,38,40],tensor_numpi:[38,40],tensorflow:[16,32,33,35,38,39,40,41],term:34,termin:[30,38],test:9,text:[9,25,27,30,34,39],textbox:30,textur:35,tf:[32,33,38,39,40,41],than:[25,34],thei:[16,19,22,32,35,37],them:[25,32,34,35,40],thi:[1,7,11,16,19,22,25,26,30,32,33,34,35,37,38,39,40,41],thing:35,think:35,those:35,though:35,thought:[34,35,37],thread:[0,2,30,38],three:[16,19,22,31,33],through:[39,40],time:[11,25,30,32,35],tinker:34,to_dataturi:38,to_embedding_model:[0,4,5,6,7,8,26,30,38,41],to_ndarrai:9,todai:35,togeth:30,together:[10,14,17,20],token:[9,25,27,34],too:35,tool:[35,41],top:[30,35,38,39,40],topk:30,torch:[1,20,21,22,32,33,38,39,40,41],torchvis:[32,38],total:9,toydata:[0,24,25,33,39,40,41],tp:1,train:[1,10,14,16,17,19,20,22,25,26,27,30,33,34,35,37,38,39,40,41],train_data:[0,2,10,11,14,17,20,25,26,30,33,34,38,39,40,41],trainabl:32,trainer:[38,40],transform:33,trigger:[38,40],trim:[34,37],triplet:[11,16,19,22,33,41],tripletdataset:[15,18,21],tripletmixin:[12,15,18,21],truncat:34,tuesdai:41,tune:[25,26,31,32,34,35,37,38,39,40,41],tuner:[0,24,26,27,30,31,32,37,38,41],tupl:[0,5,6,7,8,16,19,22,32],tutori:[39,40,41],two:[19,22,25,30,32,34,35],txt:38,type:[0,1,2,4,5,6,7,8,9,10,11,14,17,19,20,22,23,25,30,32,38,39,40],typevar:1,ui:[30,31,34,37,38,39,40],under:[31,41],underli:[38,40],underneath:38,understand:[34,37],unfold:35,unifi:34,union:[1,10,11,14,17,20,23],univers:35,unknown:[30,38],unlabel:[27,30],unlabeled_data:[26,30,41],unlik:[34,37],unlock:41,unrel:[30,33],up:[30,34,35,38,39,40,41],upsampl:9,upstream:34,uri:40,url:30,us:[1,5,6,7,8,10,14,16,17,19,20,22,25,26,31,32,33,34,35,37,38,39,40],usag:[26,41],user:[34,35,37,38,40,41],userwarn:[30,38,40],util:[21,38,40],valid:[25,35],valu:[3,9,10,14,16,17,19,20,22,23,25,30,34,35],valueerror:1,ve:[26,41],vector:[33,34,39,40,41],veri:[30,35],version:38,via:[26,31,32,33,34,37,41],video:41,view:39,vision:[32,38],visual:[38,40],vs:35,w:[9,30],wa:[35,38],wai:[25,30,33,34,35,41],wait:30,want:[32,35,38,39,40,41],warn:[38,40],we:[9,25,32,33,34,35,38,39,40,41],web:25,websit:35,wedg:[16,19,22],weight:[5,6,7,8,11,30,32,33,34,38,41],well:35,welt:25,what:[26,34,35,37,41],whatev:38,when:[3,5,6,7,8,16,19,22,25,30,33,35,37,38,41],where:[10,14,16,17,19,20,22,23,25,27,33,34,35,41],wherea:[25,30,33],whether:[25,32],which:[10,14,17,20,25,26,30,32,34,38,39,41],whole:35,why:34,wide:34,wiedersehen:25,wish:35,without:1,word:35,work:[30,32,33,34,35,38,41],world:25,worri:[26,33,41],would:35,wrap:33,write:[33,38,39,40,41],writeabl:[38,40],written:[32,33],wrong_answ:[25,39],x:[9,27,32,33,34,39],y_:33,yahaha:35,yaml:3,ye:[25,26,35,41],year:35,yet:[30,35,41],yield:[38,39,40],you:[5,6,7,8,14,17,20,25,26,30,32,33,35,38,39,40,41],your:[25,26,30,32,33,34,39,41],youtub:41,zero:35,zip:[35,38],zoo:[32,34],zoom:[35,41]},titles:["finetuner package","finetuner.helper module","finetuner.labeler package","finetuner.labeler.executor module","finetuner.tailor package","finetuner.tailor.base module","finetuner.tailor.keras package","finetuner.tailor.paddle package","finetuner.tailor.pytorch package","finetuner.toydata module","finetuner.tuner package","finetuner.tuner.base module","finetuner.tuner.dataset package","finetuner.tuner.dataset.helper module","finetuner.tuner.keras package","finetuner.tuner.keras.datasets module","finetuner.tuner.keras.losses module","finetuner.tuner.paddle package","finetuner.tuner.paddle.datasets module","finetuner.tuner.paddle.losses module","finetuner.tuner.pytorch package","finetuner.tuner.pytorch.datasets module","finetuner.tuner.pytorch.losses module","finetuner.tuner.summary module","finetuner","Data Format","One-liner <code class=\"docutils literal notranslate\"><span class=\"pre\">fit()</span></code>","Glossary","&lt;no title&gt;","&lt;no title&gt;","Labeler","Overview","Tailor","Tuner","Decisions","Philosophy","&lt;no title&gt;","Overview","Finetuning Pretrained ResNet for Celebrity Face Search","Finetuning Bi-LSTM for Question-Answering","Finetuning MLP for Fashion Image Search","Welcome to Finetuner!"],titleterms:{"1":25,Is:25,One:[26,37],advanc:30,agnost:37,all:25,answer:39,api:34,argument:[33,34],backend:34,backstori:35,base:[5,11],bi:[32,39],bidirect:33,build:[39,40],celeba:38,celebr:38,content:[0,2,4,6,7,8,10,12,14,17,20],control:30,covid:[25,33],data:[25,38,39,40],dataset:[12,13,15,18,21],decis:34,deliveri:35,design:34,displai:32,dl:34,embed:[39,40],exampl:[25,32,33],executor:3,experi:37,face:38,fashion:[25,33,40],field:25,finetun:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,35,38,39,40,41],fit:[26,30,33],flexibl:37,format:25,framework:37,glossari:27,have:25,helper:[1,13],imag:40,interact:[30,38,39,40],interfac:[30,37],jina:34,join:41,kera:[6,14,15,16],label:[2,3,25,30,34,38,39,40],last:35,learn:35,liner:[26,37],load:38,loss:[16,19,22,33],lstm:[32,33,39],manag:37,match:25,method:[30,32,33],mile:35,minimum:[34,37],mlp:[32,33,40],mnist:[25,33],model:[32,38,39,40],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],next:41,okai:25,overview:[31,37],packag:[0,2,4,6,7,8,10,12,14,17,20],paddl:[7,17,18,19],panel:30,philosophi:35,pillar:[34,37],posit:35,prepar:[38,39,40],pretrain:[32,38],progress:30,put:[38,39,40],pytorch:[8,20,21,22],qa:[25,33],question:[30,39],quick:41,relationship:34,requir:25,resnet:38,run:30,save:33,search:[38,40],simpl:[32,33],singl:34,sourc:25,start:41,step:41,submodul:[0,2,4,10,12,14,17,20],subpackag:[0,4,10],summari:[23,34],supervis:25,support:[34,41],tailor:[4,5,6,7,8,30,32,34],three:[34,37],tip:32,to_embedding_model:32,togeth:[38,39,40],toydata:9,transfer:35,tune:33,tuner:[10,11,12,13,14,15,16,17,18,19,20,21,22,23,33,34],understand:25,us:[30,41],usag:31,user:30,vgg16:32,view:30,welcom:41,why:35,without:30}})