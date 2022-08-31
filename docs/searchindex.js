Search.setIndex({docnames:["api/finetuner","api/finetuner.callback","api/finetuner.client","api/finetuner.client.client","api/finetuner.exception","api/finetuner.experiment","api/finetuner.finetuner","api/finetuner.models","api/finetuner.run","api/modules","get-started/design-principles","get-started/how-it-works","get-started/installation","index","tasks/image-to-image","tasks/text-to-image","tasks/text-to-text","walkthrough/basic-concepts","walkthrough/choose-backbone","walkthrough/create-training-data","walkthrough/index","walkthrough/integrate-with-jina","walkthrough/login","walkthrough/run-job","walkthrough/save-model"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/finetuner.rst","api/finetuner.callback.rst","api/finetuner.client.rst","api/finetuner.client.client.rst","api/finetuner.exception.rst","api/finetuner.experiment.rst","api/finetuner.finetuner.rst","api/finetuner.models.rst","api/finetuner.run.rst","api/modules.rst","get-started/design-principles.md","get-started/how-it-works.md","get-started/installation.md","index.md","tasks/image-to-image.md","tasks/text-to-image.md","tasks/text-to-text.md","walkthrough/basic-concepts.md","walkthrough/choose-backbone.md","walkthrough/create-training-data.md","walkthrough/index.md","walkthrough/integrate-with-jina.md","walkthrough/login.md","walkthrough/run-job.md","walkthrough/save-model.md"],objects:{"":[[0,0,0,"-","finetuner"]],"finetuner.callback":[[1,1,1,"","BestModelCheckpoint"],[1,1,1,"","EarlyStopping"],[1,1,1,"","EvaluationCallback"],[1,1,1,"","TrainingCheckpoint"],[1,1,1,"","WandBLogger"]],"finetuner.callback.BestModelCheckpoint":[[1,2,1,"","mode"],[1,2,1,"","monitor"]],"finetuner.callback.EarlyStopping":[[1,2,1,"","baseline"],[1,2,1,"","min_delta"],[1,2,1,"","mode"],[1,2,1,"","monitor"],[1,2,1,"","patience"]],"finetuner.callback.EvaluationCallback":[[1,2,1,"","batch_size"],[1,2,1,"","distance"],[1,2,1,"","exclude_self"],[1,2,1,"","index_data"],[1,2,1,"","limit"],[1,2,1,"","query_data"]],"finetuner.callback.TrainingCheckpoint":[[1,2,1,"","last_k_epochs"]],"finetuner.callback.WandBLogger":[[1,2,1,"","token"],[1,2,1,"","wandb_args"]],"finetuner.client":[[3,0,0,"-","client"]],"finetuner.client.client":[[3,1,1,"","FinetunerV1Client"]],"finetuner.client.client.FinetunerV1Client":[[3,3,1,"","create_experiment"],[3,3,1,"","create_run"],[3,3,1,"","delete_experiment"],[3,3,1,"","delete_experiments"],[3,3,1,"","delete_run"],[3,3,1,"","delete_runs"],[3,3,1,"","get_experiment"],[3,3,1,"","get_run"],[3,3,1,"","get_run_logs"],[3,3,1,"","get_run_status"],[3,3,1,"","list_experiments"],[3,3,1,"","list_runs"],[3,3,1,"","stream_run_logs"]],"finetuner.exception":[[4,5,1,"","FinetunerServerError"],[4,5,1,"","RunFailedError"],[4,5,1,"","RunInProgressError"],[4,5,1,"","RunPreparingError"],[4,5,1,"","UserNotLoggedInError"]],"finetuner.experiment":[[5,1,1,"","Experiment"]],"finetuner.experiment.Experiment":[[5,3,1,"","create_run"],[5,3,1,"","delete_run"],[5,3,1,"","delete_runs"],[5,3,1,"","get_run"],[5,3,1,"","list_runs"],[5,6,1,"","name"],[5,6,1,"","status"]],"finetuner.finetuner":[[6,1,1,"","Finetuner"]],"finetuner.finetuner.Finetuner":[[6,3,1,"","create_experiment"],[6,3,1,"","create_run"],[6,3,1,"","delete_experiment"],[6,3,1,"","delete_experiments"],[6,3,1,"","delete_run"],[6,3,1,"","delete_runs"],[6,3,1,"","get_experiment"],[6,3,1,"","get_run"],[6,3,1,"","get_token"],[6,3,1,"","list_experiments"],[6,3,1,"","list_runs"],[6,3,1,"","login"]],"finetuner.models":[[7,1,1,"","BERT"],[7,1,1,"","EfficientNetB0"],[7,1,1,"","EfficientNetB4"],[7,1,1,"","MLP"],[7,1,1,"","OpenAICLIP"],[7,1,1,"","ResNet152"],[7,1,1,"","ResNet50"],[7,1,1,"","SentenceTransformer"]],"finetuner.models.BERT":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.EfficientNetB0":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.EfficientNetB4":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.MLP":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.OpenAICLIP":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.ResNet152":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.ResNet50":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.models.SentenceTransformer":[[7,2,1,"","architecture"],[7,2,1,"","description"],[7,2,1,"","name"],[7,2,1,"","options"],[7,2,1,"","output_dim"],[7,2,1,"","task"]],"finetuner.run":[[8,1,1,"","Run"]],"finetuner.run.Run":[[8,6,1,"","artifact_id"],[8,6,1,"","config"],[8,3,1,"","logs"],[8,6,1,"","name"],[8,3,1,"","save_artifact"],[8,3,1,"","status"],[8,3,1,"","stream_logs"]],finetuner:[[1,0,0,"-","callback"],[2,0,0,"-","client"],[0,4,1,"","create_experiment"],[0,4,1,"","create_run"],[0,4,1,"","delete_experiment"],[0,4,1,"","delete_experiments"],[0,4,1,"","delete_run"],[0,4,1,"","delete_runs"],[0,4,1,"","describe_models"],[4,0,0,"-","exception"],[5,0,0,"-","experiment"],[6,0,0,"-","finetuner"],[0,4,1,"","fit"],[0,4,1,"","get_experiment"],[0,4,1,"","get_run"],[0,4,1,"","get_token"],[0,4,1,"","list_callbacks"],[0,4,1,"","list_experiments"],[0,4,1,"","list_model_options"],[0,4,1,"","list_models"],[0,4,1,"","list_runs"],[0,4,1,"","login"],[7,0,0,"-","models"],[8,0,0,"-","run"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","exception","Python exception"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:exception","6":"py:property"},terms:{"0":[1,13,14,15,16,17,20,23,24],"00":[14,16,20,24],"000":16,"0001":17,"003":14,"01":[17,20,23,24],"01860":14,"03":[14,16],"03544":14,"05":16,"050":[20,24],"07653":16,"08":[20,24],"0831918":15,"09":[20,24],"0994012":15,"1":[1,4,16,19,23,24],"10":[13,15,23],"104559":16,"104598":16,"108982":15,"12032":14,"128":[14,16,20,23],"1280":[7,18],"13":[14,20,24],"13469":16,"14":15,"14489":15,"15":[14,16],"15746":16,"16":14,"16603":14,"1e":[14,15,16,23],"2":[1,13,15,20,21,23,24],"20":[1,10],"202":[14,16],"204":[14,16],"2048":[7,18],"207":[14,16],"214":[20,24],"217":[20,24],"218":[14,16],"219":15,"222":15,"224":14,"225":[14,16],"228":[20,24],"23":[20,24],"231":[14,16],"231756":15,"232":[20,24],"233":[14,15,16],"23632":14,"238":[20,24],"240":[15,20,24],"241773":15,"246":15,"248":[14,15],"253423":15,"256587":15,"288791":15,"3":[12,13,16,23],"32":16,"33912":16,"346108":15,"350172":15,"35847":15,"36":16,"37":16,"37209":14,"38":15,"39":14,"4":[0,6,23],"40":[13,16],"41":[14,16],"415924":15,"45":10,"487505":15,"5":[0,6,14,15,16,20],"50":[20,24],"5000":16,"51000":21,"512":23,"539948":15,"54":[20,24],"6016":14,"611976":15,"62972acb5de25a53fdbfcecc":[20,24],"62a1af491597c219f6a330f":15,"62b33cb0037ad91ca7f20530":14,"62b9cb73a411d7e08d18bd16":16,"64":[0,6],"7":[12,13],"76":14,"768":[7,18,21],"8":1,"818":16,"856287":15,"9":21,"902417":15,"94393":16,"95728":16,"96686":16,"97529":16,"99301":16,"99720":16,"case":[7,10,13,15,16,18,20,23],"class":[1,3,5,6,7,8,11],"default":[0,1,6,11,14,15,16,17,21,23],"do":[1,10,11,14,15,16,18,20],"final":18,"float":[0,1],"function":[0,1,7,8,23],"import":[14,15,16,17,18,19,20,21,22,23,24],"int":[0,1,3,7],"new":[3,13,20],"public":13,"return":[0,3,5,6,8,16,21],"short":[16,18],"super":21,"true":[0,1,6,7,16],"while":[1,10,11,14,15,16,23],A:[0,1,3,5,6,7,8,17,22,23],As:[10,14,16,19,21],At:[0,11,19],By:23,For:[1,3,11,14,15,16,17,19,23],If:[0,1,3,6,10,16,17,19,21,23,24],In:[1,10,15,16,21,24],It:[11,18,19,22,24],On:[11,21],Such:8,That:[11,14,15,16],The:[0,1,3,7,11,13,14,15,16,18,19,21],Then:15,There:10,These:[0,16],To:[1,16,18,23],With:13,__main__:[14,15,16,20,24],_basecli:3,_modelstub:7,abl:[10,14,15],about:[10,13,15,16],abov:[19,23],absolut:1,accept:19,access:[13,16],accomplish:16,accord:1,account:[1,6],across:[0,1,6,10],action:[14,15],activ:[7,13],actual:21,ad:21,adadelta:[0,23],adagrad:[0,23],adam:[0,6,16,17,23],adamax:[0,23],adamw:[0,23],adapt:20,add:[7,21],addit:[0,23],addition:[14,15],address:20,adjust:11,advanc:20,after:[0,1,6,14,15,16,22],again:11,against:[10,16],ai:13,algorithm:11,all:[0,1,3,6,10,11,13,14,15,23],allow:23,almost:19,alreadi:[14,15,16],also:[10,14,15],amount:16,an:[0,1,3,4,5,6,8,13,15,16,17,21,23,24],analysi:10,anchor:[11,16,23],angularloss:0,angularmin:[0,23],ani:[0,7,11,18,21,23],anoth:[14,16,19],apach:13,apart:16,api:[3,5,8],app:[14,15,16,20,24],appli:[7,11,13,16,18,23],applic:[0,13],approach:[11,14],ar:[0,1,10,13,14,15,16,17,18,19,20,22,23],arcfaceloss:0,architectur:[7,11,18],argument:[0,1,5,14,15,16,17,21],artifact:[8,11,13,14,15,16,20,21],artifact_id:[8,21,24],asgd:[0,23],assign:19,attach:[0,23],attempt:20,attribut:[16,23],authent:1,auto:1,automat:21,avail:[0,1,3,11,14,15,16,19,23],average_precis:15,awai:[11,14],await:[14,15,16],b0:23,back:[11,13],backbon:[10,11,20,23],backend:10,background:10,base:[1,3,4,5,6,7,8,10,15,16,17,18],baselin:1,basemetriclossfunct:0,basemin:0,basesubsetbatchmin:0,basetuplemin:0,basic:20,batch:[0,1,6,23],batch_siz:[0,1,6,14,16,20,23],batcheasyhardmin:0,batchhardmin:0,becaus:[19,20,24],becom:16,been:[16,18,23],befor:[10,14,15,22],being:21,belong:[11,16],below:[14,15],benchmark:14,bert:[7,17,18,21],best:[1,14,15],bestmodelcheckpoint:[1,14,15],better:[20,21],between:[16,23],beyond:23,bia:7,bias:1,block:19,bookcorpu:[7,18],bool:[0,1,7],boost:10,both:16,bring:20,browser:22,build:13,busi:20,c:20,calcul:[1,23],calendar:13,call:[0,6,14,15,16,18,21,23],callback:[0,6,9,14,15,16],callbackstubtyp:0,can:[0,1,10,11,13,14,15,16,17,18,20,21,23,24],candid:23,caption:15,care:10,carri:19,cat:11,catalog:1,centroidtripletloss:0,certain:[11,20],challeng:14,chang:[1,23],channel:13,chat:13,check:[1,8,12,14,15,16,20,23],checkpoint:1,choos:[0,10,14,15,16,23],chosen:16,chunk:[15,19],circleloss:0,classif:[10,11],classifi:11,click:21,client:[0,1,5,6,8,9],clip:[7,10,18,23],clipimageencod:15,cliploss:[0,15,23],cliptextencod:15,close:[16,24],closer:[14,16],cloud:[8,13,14,15,16,20,21,22,24],cnn:[7,18],code:[4,14,15,17,19],collat:8,common:16,commun:13,compar:10,compat:23,complet:[16,22],complex:[11,13,20],comput:[1,11,16,20],concept:20,conduct:10,config:[5,8],configur:[3,7,8,11,17,20],connect:24,consecut:1,consid:1,consist:[11,14,16],construct:[0,14,15,23],constructor:23,contain:[8,11,15,16,17,21],content:[9,10,19],context:[14,16],continu:[14,15],contrastiveloss:0,conveni:21,convert:18,copi:21,correspond:[11,14],cosfaceloss:0,cosin:1,could:[8,16],count:1,cours:16,cpu:[0,3,6,11,14,15,16,20,23],creat:[0,1,3,5,6,11,14,15,16,19,21,23,24],create_experi:[0,3,6,23],create_run:[0,3,5,6],created_at:[5,8],creation:[5,8],cross:[10,13,19],crossbatchmemori:0,cumul:10,current:[0,1,6,14,15,17],d:16,da:[14,16,21],dai:15,data:[0,1,5,10,11,20,22,23,24],dataclass:16,dataload:0,dataset:[14,15,17,18,20,23,24],dcg_at_k:15,debug:[14,16],decai:23,decid:[11,16],decis:1,declar:17,dedic:11,deep:[13,20],defin:[1,17],definit:1,delet:[0,1,3,5,6],delete_experi:[0,3,6],delete_run:[0,3,5,6],deliv:[13,20],demonstr:14,dens:10,depend:[0,10,24],describ:0,describe_model:[0,10,14,15,18],descript:[0,3,5,6,7,8,14,15,16,18,23],descriptor:18,detail:[0,4,14,15,16,23],detect:10,develop:23,devic:3,dict:[0,1,3,7,8,16],did:[15,21],differ:[0,10,11,14,16,17,20,22,23],dimens:0,dimension:[13,16],directli:[8,14,15],directori:[0,6,8,17,19,20],discount:10,discov:19,discuss:[13,16],dissimilar:16,distanc:[1,10,23],distanceweightedmin:0,distilbert:[7,16,18],distribut:[13,20],doc:[1,14,21],docarrai:[1,16,19,20,21,23],docker:21,document:[0,3,10,11,14,15,16,19,21,23],documentarrai:[0,1,5,10,11,16,19,20,22,23],doe:[14,15],doesn:1,dog:11,domain:[13,20],don:[10,14,15],done:[11,14,15,16,20,24],download:[8,14,15,24],dump:8,duplic:16,dure:[11,16,19,23],e:[1,11,14,15,19,20,23],each:[1,7,11,14,15,16,17,19],earlystop:1,easi:[13,16,20,21],easier:13,easili:[13,14,16],ecosystem:[0,11,13,14,15,16,20,21,22,24],effect:13,effici:20,efficientnet:[18,23,24],efficientnet_b0:[7,18,23],efficientnet_b4:[7,18],efficientnetb0:7,efficientnetb4:7,either:[0,1,3,5,14,15,20],emb:[16,20],embed:[1,14,16,18,20,21,23],embeddingregularizermixin:0,embeddingsalreadypackagedastriplet:0,empti:16,enabl:[16,20,23],encod:[7,15,20,21],end:[1,14,15,16],endpoint:21,engin:[11,13,20],english:[7,18],enough:11,ensur:16,entri:3,env:21,epoch:[0,1,6,14,15,16,20,23],error:4,etc:1,euclidean:1,eval:[14,15],eval_data:[0,6,14,15,23],evalu:[0,1,19,23,24],evaluationcallback:[1,14,15,16],evaul:15,event:[3,13],everi:[0,1,5,6,13],everyth:15,exampl:[1,11,14,15,16,17,21,23,24],except:[0,9],exclud:1,exclude_self:1,execut:[1,13,14,15,16],executor:20,exist:24,expect:[0,11,14],experi:[0,3,6,8,9,10,13,16,17,20,21,23,24],experiment_nam:[0,3,6,8,16,17,21,23],explain:11,explicitli:23,explor:16,expos:21,extra:16,f1_score_at_k:15,f:[16,21,23,24],face:16,factor:10,factori:1,fail:23,fals:[0,6,7,14,15,16,23],fanci:10,fashion:15,fast:18,fastaploss:0,faster:13,featur:[11,13],few:21,field:16,filter:23,find:16,fine:[0,5,7,8,10,13,17,18,19,20,22,23,24],finetun:[10,11,12,16,17,18,19,20,21,22,24],finetuner_label:[11,16,19],finetunerexecutor:8,finetunerservererror:4,finetunerv1cli:[3,5,8],finish:[8,14,15,16,20,21,23,24],fintun:23,first:[1,14,15],fit:[0,1,16,17,19,20,23],flexibl:[10,11,23],flickr8k:23,flickr:[23,24],flow:21,follow:[14,16,19,20,24],form:19,found:[0,16],framework:23,free:13,freez:[0,6,11,13,23],friendli:[14,15],from:[7,8,11,14,15,16,18,19,20,21,22,23],full:[14,15],fulli:20,further:16,furthermor:14,futur:16,g:[11,19,23],gain:[10,16],gener:[17,18],generalizedliftedstructureloss:0,genericpairloss:0,geologist:16,get:[0,3,5,6,8,14,15,18,20,23],get_experi:[0,3,6,24],get_run:[0,3,5,6,14,15,16,21,24],get_run_log:3,get_run_statu:3,get_token:[0,6,21],give:[10,11,23],given:[3,10],go:[11,14,21],gonna:14,good:16,gpu:[0,3,11,13,20,21],great:16,grpc:21,guid:[14,15,16],ha:[1,14,15,16,18,23],had:14,hand:[11,13,21],handl:13,happen:[14,15],hard:[13,23],hardwar:20,have:[1,10,11,12,13,14,15,16,18,20,21,22,23,24],hdcminer:0,head:[0,11,18],height:14,help:10,here:[13,14,15,21,23],hidden:[7,11],hidden_s:7,high:[16,18,23],highest:16,highli:24,hire:13,hit_at_k:15,hold:15,homogen:16,host:16,hour:[14,15,16],how:[14,15,16,24],howev:[13,15,16,24],http:19,hubbl:[0,5,6,14,15,16,20,24],hug:16,huggingfac:18,human:14,hyper:[10,15,23,24],i:[1,14,15,16,20],ical:13,id:[8,14,15,16,20,21,24],idea:13,ie:16,imag:[7,10,11,18,19],imagenet:[7,18],importantli:16,improv:[1,10,11,13,16],includ:[0,1,18,20],index:[1,10,13,14,16],index_data:[1,14,15,16],indic:[0,8,16],infer:18,info:[14,15,20,24],inform:[15,16,20],infrastructur:[13,20],init:1,initi:[6,14,15,21],input:[7,21],input_s:7,insid:[3,5,19,23],instal:1,instanc:[16,23],instant:13,instead:11,integ:1,integr:[13,14,15,16,20],interest:20,interfac:20,intern:10,interpret:11,interv:0,intrapairvarianceloss:0,involv:16,ir:1,irrelev:[16,20],isol:11,item:0,iter:[3,8],its:[0,3,5,6,8,14,15,16,23],jina:[0,11,13,14,15,16,20,21,22,24],jinahub:21,job:[11,14,17,20,22,24],journei:[14,15],jtype:21,just:[16,21],k:[1,10],keep:[8,16],kei:1,kept:1,keyword:[1,5],know:[10,14,15],knowledg:[11,20],kwarg:5,l2:7,label:[11,16,19],lack:20,larg:[16,20],largemarginsoftmaxloss:0,last:[0,1,7],last_k_epoch:1,lastli:[14,16],later:11,latest:[12,13],layer:[0,7,11,13],lbfg:[0,23],lead:16,learn:[0,10,13,14,16,17,19,20,23],learning_r:[0,6,14,15,16,23],left:14,length:16,less:1,let:[10,14,15,16],level:11,leverag:[11,20,22],librari:[10,18],licens:13,liftedstructureloss:0,light:18,like:[1,14,16,17,20,23],likewis:16,limit:1,link:[3,22,23],linux:12,list:[0,1,3,5,6,7,14,15,17,18],list_callback:0,list_experi:[0,3,6],list_model:0,list_model_opt:0,list_run:[0,3,5,6],live:13,ll:[0,6,14,15,23],load:[16,18,21],load_uri_to_image_tensor:14,local:[14,15,19,21,23],log:[1,3,8,11,14,15,16,20,22,23,24],log_entri:[20,24],logger:1,login:[0,6,11,14,15,16,20,21,24],look:[0,6,11,14,16,23],loop:1,loss:[0,1,6,13,14,15,16,20,23,24],lot:[10,16],lr:23,mac:12,machin:[10,11,14,15,17,20],made:[1,16],mai:16,mainstream:13,maintain:20,make:[1,10,11,12,13,16],manag:13,mani:[13,16],marco:[7,18],margin:23,marginloss:0,match:[1,16],matter:20,max:1,maxim:1,maximumlossmin:0,mean:10,measur:16,meet:13,member:13,memori:11,messag:4,metadata:8,method:20,metric:[0,1,10,14,15,16,19,23],might:[14,15,16,24],min:1,min_delta:1,mine:13,miner:[0,6,20],miner_opt:[0,6,23],minim:1,minimum:1,mlp:[7,23],mnt:21,modal:[10,13,19],mode:1,model:[0,1,5,6,8,9,10,13,17,20,22,23,24],model_average_precis:[14,16],model_dcg_at_k:[14,16],model_f1_score_at_k:[14,16],model_hit_at_k:[14,16],model_ndcg_at_k:[14,16],model_opt:[0,6,23],model_precision_at_k:[14,16],model_r_precis:[14,16],model_recall_at_k:[14,16],model_reciprocal_rank:[14,16],modifi:23,modul:[9,13],monitor:1,month:13,more:[10,13,14,16,18,20,21,23],most:[1,15,16],mount:21,mrr:10,ms:[7,18],msmarco:[7,16,18],much:16,multimod:16,multipl:[15,16],multipleloss:0,multisimilarityloss:0,multisimilaritymin:0,nadam:[0,23],name:[0,3,5,6,7,8,14,15,16,20,23],ncaloss:0,ndcg:10,ndcg_at_k:15,necessari:6,need:[10,11,14,15,18,19,20,21],neg:[11,13,14,16,23],network:14,neural:[10,13],next:[13,16],nlp:20,non:[13,16,20],none:[0,1,3,5,6,7],normal:[7,10],normalizedsoftmaxloss:0,note:[6,17,18,23],now:[14,15,16,23,24],npairsloss:0,ntxloss:0,num_work:[0,6,23],number:[0,1,3,14,15,16],object:[0,1,5,6,8,10,11,19],observ:10,obtain:22,occur:4,older:1,onc:[11,16,18,20,21,22,24],one:[0,1,11,13,15,16,18,23],ones:16,onli:[1,10,11,14,15,19,21,22,23],openai:[7,10,15,18,19],openaiclip:7,opensourc:13,optim:[0,6,16,17,19,20],optimis:13,optimizer_opt:[0,6,23],option:[0,1,3,5,6,7,8,10,14,17,21,22,23],order:[14,16,21],organ:[10,17],other:[1,10,11,13,14,15,16,21,24],otherwis:1,our:[3,11,13,14,15,16,23],out:[10,20,23],output:[0,7,11,16],output_dim:[0,6,7,18,23],over:[1,19,20],overwrit:1,own:20,owner:20,packag:9,page:21,pair:[7,14,16,18,19],pairmarginmin:0,parallel:19,param:[0,3,6],paramet:[0,1,3,5,6,7,8,10,11,20,23,24],parent:15,part:21,pass:[0,1,16,23],patch32:[7,15,18],patch:14,path:[8,21],patienc:1,penal:23,pencil:19,per:0,percept:14,perfect:24,perform:[10,13,16,17,18,20],perhap:16,person:22,perspect:11,pick:14,pictur:15,piec:[14,15,17],pip:[1,12,13],pleas:[3,14,16,17,20,21,23],plu:20,plug:15,png:19,pool:[11,18],pop:22,popular:16,port:21,posit:[11,14,16,23],possibl:23,post:21,power:13,practic:20,pre:[13,16,20],preced:1,precis:10,precision_at_k:15,predict:11,prepar:[11,14,15,20,23],preprocess:8,pretrain:[7,13,18],previous:[14,15,24],print:[14,15,16,20,21,23,24],probabl:11,problem:20,process:[16,23],produc:[10,18],product:[13,15,20],profil:22,progress:23,project:0,promis:13,proper:11,properti:[5,8],protocol:21,provid:[1,10,14,15,16,18,20,21,23],provis:13,proxim:16,proxyanchorloss:0,proxyncaloss:0,prune:13,pull:[8,11,14,16,21],purpos:[16,19],push:[0,5,11,14,15,16,20,22,24],py:[14,15,16,20,24],python:[12,13],pytorch:[0,16,23],qa:17,qualifi:1,qualiti:20,quantiti:1,queri:[1,10,13,14,16],query_data:[1,14,15,16],question:[13,16],queue:23,quora:[16,17],quora_index_dev:16,quora_query_dev:16,quora_train:16,quoraqa:17,r_precis:15,radam:[0,23],random:17,rank:[10,16],rate:[0,17,23],re:[14,17,19],readi:[13,14,15,16,20],recal:10,recall_at_k:15,recent:1,recip:10,reciprocal_rank:15,recommend:[14,19],reconnect:[14,15,16,24],reduc:[14,23],reduct:13,refer:[20,23],relev:16,reli:1,relu:7,remov:[11,18],repres:[8,10,11,16],represent:[7,14,16],reproduc:14,request:[3,5,8],requir:[14,15,16,18,20,21,22],resnet152:[7,18],resnet18:17,resnet50:[7,17,18,20],resnet:[14,17,18],resourc:[11,13,22],respect:16,respons:16,result:[1,14,15,16,20],retriev:[14,15,16],returned_doc:21,right:[14,21],rmsprop:[0,23],root:19,rprop:[0,23],rtype:[0,3,6],run1:17,run2:17,run3:17,run:[0,3,5,6,9,13,14,15,16,17,20,21,24],run_config:3,run_nam:[0,3,5,6,14,15,16,17,21,23],runfailederror:4,runinprogresserror:4,runpreparingerror:4,s:[0,11,14,15,16,19,21,23],same:[11,14,16,19],sampl:16,save:[1,8,19,20,21],save_artifact:[8,14,15,16,20,21,24],scale:20,schedul:0,scheduler_step:[0,6,23],scheme:19,scratch:7,search:[1,13,18,19,20],second:13,section:[11,14,15,16],see:[0,13,14,15,20,21,22,23,24],select:[16,23],self:1,sell:19,send:[5,8,23],sens:16,sent:23,sentenc:[7,16,17,18],sentencetransform:7,sentiment:10,separ:[16,19],server:8,serversentev:3,session:24,set:[0,1,7,14,16,20,21,23],sever:[10,14,15,16,18,24],sgd:[0,17,23],shape:[14,21],shift:20,shirt:19,should:[0,1,16,18,19,20,22,23],show:[1,12,16,24],showcas:15,shown:[14,19],side:21,sigmoid:7,signaltonoiseratiocontrastiveloss:0,signific:10,similar:[10,14,16],similarli:21,simpl:[7,13,16,20,21,22],simplest:23,sinc:[14,15,16,22],singl:19,size:[1,7,16,19,23,24],skirt:19,slack:13,slim:19,so:[1,10,16],softtripleloss:0,solut:[13,20],some:[14,15,16,21],someth:23,sota:[13,20],sourc:[0,1,3,4,5,6,7,8],space:16,sparseadam:[0,23],specif:[0,11,13,14,17,21,23],specifi:[0,3,6,16,21,23],speed:14,spherefaceloss:0,spin:11,sqeuclidean:1,src:[14,15,16,20,24],start:[0,14,15,16,17,21,22,23,24],state:[1,10],statu:[3,5,8,14,15,16,23,24],step:[0,20],still:1,stop:1,storag:[8,10,11,21,23],store:[8,14],str:[0,1,3,5,6,7,8,16],stream:[3,8,13,24],stream_log:[8,20,24],stream_run_log:3,streamlin:13,string:[0,7,8],strip:19,structur:16,stub:[0,7],submit:[14,15,16,20],submodul:9,subpackag:9,subscrib:13,successfulli:[14,15,16,20,22,23,24],suggest:10,suitabl:18,summari:16,supconloss:0,supervis:19,supplementari:16,suppli:17,support:[15,16,18],sure:[1,12,13,16],system:10,t:[1,10,14,15,19,20],tabl:[0,15],tackl:10,tag:[16,19],take:[14,15,16,23,24],talk:13,tanh:7,task:[7,11,13,17,18,20],tensor:16,tereshkova:[20,23],termin:[20,22,23,24],test:14,text:[7,10,18,19,21],textual:15,than:[1,18],thei:16,them:23,thi:[0,1,8,10,11,14,15,16,17,20,21,22,23,24],thing:16,three:[10,16,21],through:16,thu:20,time:[5,8,11,15,19,21,24],titl:15,tll:14,togeth:[11,16],token:[0,1,21,22],top:[1,18,21,23],torchvis:18,total:14,train:[0,1,5,7,10,13,14,15,16,17,18,20,22,23,24],train_da:19,train_data:[0,5,6,14,15,16,20,23],train_loss:1,trainingcheckpoint:1,transform:[7,16,17,18,19],transmiss:14,trial:23,tripl:23,triplemarginloss:23,triplet:14,tripletloss:14,tripletmarginloss:[0,6,14,23],tripletmarginmin:[0,23],trivial:[13,20],tuesdai:13,tune:[0,5,7,8,10,13,17,18,19,20,22,23,24],tuned_model:[20,24],tuner:1,tupl:[0,16],tupletmarginloss:0,turn:11,tutori:13,two:[14,15,16,17,19],type:[0,1,3,5,6,8,16],u:[12,13],under:[13,21],understand:[14,15],uniformhistogrammin:0,union:[0,1,5],uniqu:[10,16],unknown:4,until:23,unzip:21,up:[11,13,14,15,16,20,22,23],uplift:13,uri:19,us:[0,1,3,6,8,14,15,16,17,18,20,21,23],user:[0,10,14,15,22],usernotloggedinerror:4,uses_with:21,usr:[14,15,16,20,24],usual:[20,23],v0:21,v1:3,v3:[7,16,18],val_loss:1,valid:[0,1],valu:[1,14,15,16,21,23],variant:15,variou:10,ve:[16,21],vector:[10,11],veri:16,version:12,via:[13,21,23],vicregloss:0,video:13,view:0,vigil:[20,23],visibl:[11,16,22],vision:[19,20],visit:[3,16],vit:[7,15,18],volum:21,wa:[10,19],wai:[10,13,14,15,16,19,21],wandb:1,wandb_arg:1,wandblogg:1,want:[8,15,16,23],we:[0,6,10,11,13,14,15,16,19,21,24],websit:16,weight:[1,13,18,23],weight_decai:23,weightregularizermixin:0,well:16,what:[10,14,15,16,21],when:[1,13,14,17,19,20,21],where:[8,11,13,14,16,18,23],whether:[0,1,7],which:[0,1,10,11,14,15,16,19,20,21],who:20,why:20,wide:18,width:14,wikipedia:[7,18],wild:17,window:[12,22],within:16,without:[13,23],work:[14,17,23],worker:0,workflow:13,worri:[10,13],wrap:19,yaml:21,yet:13,yield:[3,8],you:[0,1,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],your:[0,11,12,13,14,15,16,17,19,20,21,22,23,24],youtub:13,zip:21,zoom:13},titles:["finetuner package","finetuner.callback module","finetuner.client package","finetuner.client.client module","finetuner.exception module","finetuner.experiment module","finetuner.finetuner module","finetuner.models module","finetuner.run module","finetuner","Design Principles","How Does it Work?","Installation","Welcome to Finetuner!","Image-to-Image Search via ResNet50","Text-to-Image Search via CLIP","Text-to-Text Search via BERT","Basic Concepts","Backbone Model","Prepare Training Data","Walkthrough","Integration","Login","Run Job","Save Artifact"],titleterms:{"1":11,"2":11,"do":22,about:14,advanc:23,also:16,an:11,artifact:24,backbon:[14,15,16,18],base:11,basic:17,bert:16,callback:1,client:[2,3],clip:[15,19],cloud:[11,23],concept:17,configur:23,construct:11,content:[0,2],contrast:11,convert:11,data:[14,15,16,19],dataset:16,dedic:10,design:10,documentarrai:21,doe:11,easi:10,emb:21,embed:[10,11],evalu:[14,15,16],except:4,executor:21,experi:5,explain:19,fine:[11,14,15,16,21],finetun:[0,1,2,3,4,5,6,7,8,9,13,14,15,23],finetunerexecutor:21,fit:[14,15],fly:11,focu:10,how:11,i:22,imag:[14,15],info:16,instal:[12,13],integr:21,job:23,join:13,learn:11,login:22,metric:11,miner:23,model:[7,11,14,15,16,18,19,21],modul:[0,1,2,3,4,5,6,7,8],monitor:[14,15,16],need:22,optim:[10,23],packag:[0,2],paramet:[14,15,21],prepar:19,principl:10,qualiti:10,resnet50:14,run:[8,23],save:[14,15,16,24],search:[10,14,15,16],see:16,step:11,submit:23,submodul:[0,2],subpackag:0,support:13,task:[10,14,15,16],text:[15,16],train:[11,19],triplet:11,tripletmarginloss:16,tripletmarginmin:16,ttl:14,tune:[11,14,15,16,21],us:[10,13],via:[14,15,16],walkthrough:20,welcom:13,why:22,work:11,your:10}})