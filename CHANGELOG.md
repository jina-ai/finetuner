









# Changelog

The changes of each release is tracked by this file.

<a name=release-note-0-0-2></a>
## Release Note (`0.0.2`)

> Release time: 2021-10-04 16:33:10



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  🙇


### 🍹 Other Improvements

 - [[```5a7e2ae8```](https://github.com/jina-ai/finetuner/commit/5a7e2ae87a2a3d09212453431f5884e149dca873)] __-__ __cd__: add release script to cd (*Han Xiao*)

<a name=release-note-0-0-3></a>
## Release Note (`0.0.3`)

> Release time: 2021-10-19 23:04:12



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Wang Bo,  Maximilian Werk,  Tadej Svetina,  Alex Cureton-Griffiths,  Roshan Jossy,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```84585bee```](https://github.com/jina-ai/finetuner/commit/84585bee27cfbc1de9ea1efd6e1345215c888a5c)] __-__ refactor head layers (#130) (*Tadej Svetina*)
 - [[```b624a62a```](https://github.com/jina-ai/finetuner/commit/b624a62a9006bace7da72eeb4fe1dcae545ce803)] __-__ __tuner__: allow adjustment of optimizer (#128) (*Tadej Svetina*)
 - [[```98c584e4```](https://github.com/jina-ai/finetuner/commit/98c584e4cde94d65782d01e5cf3cfc169b06aa25)] __-__ enable saving of models in backend (#115) (*Maximilian Werk*)
 - [[```2296ca09```](https://github.com/jina-ai/finetuner/commit/2296ca0999acc43a86515921626f3c870fbb29f4)] __-__ __tuner__: add gpu for paddle and tf (#121) (*Wang Bo*)
 - [[```c60ec838```](https://github.com/jina-ai/finetuner/commit/c60ec838666cebac8994687e15da99eb10efda60)] __-__ __tuner__: add gpu support for pytorch (#122) (*Tadej Svetina*)
 - [[```c971f824```](https://github.com/jina-ai/finetuner/commit/c971f824dd8b4074da4e58cdf1a038d3142a059e)] __-__ logging of train and eval better aligned (#105) (*Maximilian Werk*)
 - [[```a6d16ff2```](https://github.com/jina-ai/finetuner/commit/a6d16ff2976eebeb38c7042365700d24b27fb5ad)] __-__ __tailor__: add display and refactor summary (#112) (*Han Xiao*)
 - [[```bd4cfff4```](https://github.com/jina-ai/finetuner/commit/bd4cfff4fc73760ad8087f0a0968dac3a4efef1d)] __-__ __fit__: add tailor to top-level fit function (#108) (*Han Xiao*)
 - [[```82c2cc8d```](https://github.com/jina-ai/finetuner/commit/82c2cc8dc8d1eedc2d8dde9ad3b8f33f64c6e21e)] __-__ __tailor__: attach a dense layer to tailor (#96) (*Wang Bo*)
 - [[```04de292a```](https://github.com/jina-ai/finetuner/commit/04de292abf47dd17252c3d2833ec785dcae5d164)] __-__ __tailor__: add high-level framework-agnostic convert (#97) (*Han Xiao*)

### ⚡ Performance Improvements

 - [[```68fc7839```](https://github.com/jina-ai/finetuner/commit/68fc7839a174078e9623c7a5cf6f7c5ad7f61c2f)] __-__ __tuner__: inference mode for torch evaluation (#89) (*Tadej Svetina*)

### 🐞 Bug fixes

 - [[```b951426d```](https://github.com/jina-ai/finetuner/commit/b951426d4af1dcdb69ef1dc924aa9aae5e7a6bba)] __-__ change helper function to private (#151) (*Han Xiao*)
 - [[```bc8b36ef```](https://github.com/jina-ai/finetuner/commit/bc8b36efa5b98c44eb8cb34ce5aa4798e87d0f4a)] __-__ __demo__: fix celeba example docs, logic, code (#145) (*Han Xiao*)
 - [[```ed6d8c67```](https://github.com/jina-ai/finetuner/commit/ed6d8c67c49567c6fc6f4ad61a7e754894873a14)] __-__ frontend layout tweaks (#142) (*Han Xiao*)
 - [[```02852803```](https://github.com/jina-ai/finetuner/commit/028528033f42bfdd3a1338e7cc1b8de9e42bbc9c)] __-__ overfit test (#137) (*Tadej Svetina*)
 - [[```5a25a729```](https://github.com/jina-ai/finetuner/commit/5a25a7296a4eacb1a9b6cd05345e8e866d364a8c)] __-__ __helper__: add real progressbar for training (#136) (*Han Xiao*)
 - [[```5196ce2a```](https://github.com/jina-ai/finetuner/commit/5196ce2a65905f7599c0b3cd68f64303dfae3002)] __-__ __api__: add kwargs to fit (#95) (*Han Xiao*)
 - [[```1a8272ca```](https://github.com/jina-ai/finetuner/commit/1a8272ca07dade9c265c6080f341049d80374d40)] __-__ threading also for gateway (#83) (*Maximilian Werk*)
 - [[```e170d95b```](https://github.com/jina-ai/finetuner/commit/e170d95bcf2e07cc6536c16efac07d88e3d9bcc5)] __-__ __cd__: fix prerelease script (*Han Xiao*)

### 🧼 Code Refactoring

 - [[```2916e9f5```](https://github.com/jina-ai/finetuner/commit/2916e9f5393a1ce3d05dc1887744c76f2b2f77c0)] __-__ __tuner__: revert some catalog change before release (#150) (*Han Xiao*)
 - [[```635cd4c2```](https://github.com/jina-ai/finetuner/commit/635cd4c20a197f3fc1c63aba4989ff9d012430e8)] __-__ adjust type hints (#149) (*Wang Bo*)
 - [[```b67ab1a5```](https://github.com/jina-ai/finetuner/commit/b67ab1a5a9300e156a68588e51391a8e39aeaae3)] __-__ __helper__: move get_tailor and get_tunner to helper (#134) (*Han Xiao*)
 - [[```052adbb2```](https://github.com/jina-ai/finetuner/commit/052adbb2cb7256435599f113adfaa1f2dec2666c)] __-__ __helper__: move get_tailor and get_tunner to helper (#131) (*Han Xiao*)
 - [[```d8ff3a5b```](https://github.com/jina-ai/finetuner/commit/d8ff3a5b875eeb31d46da2feeb8690469d9384c3)] __-__ __labeler UI__: js file into components (#101) (*Roshan Jossy*)
 - [[```80b5a2a1```](https://github.com/jina-ai/finetuner/commit/80b5a2a16678794bb322f430db73cea5cfbf37c3)] __-__ __tailor__: rename convert function to_embedding_model (#103) (*Han Xiao*)
 - [[```c06292cb```](https://github.com/jina-ai/finetuner/commit/c06292cbc502386a51f473bb84d74bc36f0e29c2)] __-__ __tailor__: use different trim logic (#100) (*Han Xiao*)
 - [[```1956a3d3```](https://github.com/jina-ai/finetuner/commit/1956a3d3ce24a58497dfd799cee8be15161b52b6)] __-__ __tailor__: fix type hint in tailor (#88) (*Han Xiao*)
 - [[```91587d88```](https://github.com/jina-ai/finetuner/commit/91587d8884c4e34b6f1e1b5fa1b11ed43ca31253)] __-__ __tailor__: improve interface (#82) (*Han Xiao*)
 - [[```56eb5e8f```](https://github.com/jina-ai/finetuner/commit/56eb5e8f549bc7877aa6804567356348d4896ab4)] __-__ __api__: move fit into top-most init (#84) (*Han Xiao*)

### 📗 Documentation

 - [[```c2584876```](https://github.com/jina-ai/finetuner/commit/c2584876d07fee42169ab86f711b4012af928e9f)] __-__ add catalog to docs (#147) (*Maximilian Werk*)
 - [[```6fd3e1ea```](https://github.com/jina-ai/finetuner/commit/6fd3e1ea63ced61c32d1a12e4cb2d4ba76a8d870)] __-__ __tuner__: add docstrings (#148) (*Tadej Svetina*)
 - [[```177a78dd```](https://github.com/jina-ai/finetuner/commit/177a78ddb5097cd3df7f4f1dc119ef101a1816ea)] __-__ fix generate docs (#144) (*Maximilian Werk*)
 - [[```ac2d23de```](https://github.com/jina-ai/finetuner/commit/ac2d23dec56d17c8aee7b46703e80fe26c227d42)] __-__ polish (#146) (*Alex Cureton-Griffiths*)
 - [[```b0da1bf6```](https://github.com/jina-ai/finetuner/commit/b0da1bf69b66de8e078b72d4c4f73d58b6ce2fca)] __-__ add celeba example (#143) (*Wang Bo*)
 - [[```475c1d8b```](https://github.com/jina-ai/finetuner/commit/475c1d8b9203f45a4c68102e3d30503511856b46)] __-__ __tuner__: add loss function explain for tuner (#138) (*Han Xiao*)
 - [[```f47e27a3```](https://github.com/jina-ai/finetuner/commit/f47e27a3a94d7f5536bd48fc530235522353567f)] __-__ update banner hide design (*Han Xiao*)
 - [[```11a6a8b9```](https://github.com/jina-ai/finetuner/commit/11a6a8b9abd78bf17d523af25b33074ec72e67f8)] __-__ add interactive selector (*Han Xiao*)
 - [[```08ba5e06```](https://github.com/jina-ai/finetuner/commit/08ba5e0628f5219fa21e1304d940b195cce7d84f)] __-__ add tailor feature image (*Han Xiao*)
 - [[```528c80d5```](https://github.com/jina-ai/finetuner/commit/528c80d5211bd2db9f4833ab9c6c1ead4a8df556)] __-__ __tailor__: add docs for tailor (#119) (*Han Xiao*)
 - [[```04c22f74```](https://github.com/jina-ai/finetuner/commit/04c22f7491458a08f203ed4964dd2e45248799f3)] __-__ __tailor__: add first draft on tailor (*Han Xiao*)
 - [[```e62f77ea```](https://github.com/jina-ai/finetuner/commit/e62f77ea5e8c060aa2d0c346d79b7fa977256dbd)] __-__ __helper__: add docstring for types (#98) (*Han Xiao*)

### 🏁 Unit Test and CICD

 - [[```6b8eca8c```](https://github.com/jina-ai/finetuner/commit/6b8eca8c212c27344e3b7fbb9d762dc71b1df387)] __-__ use jina git source as test dependencies (#135) (*Han Xiao*)
 - [[```f91f39f5```](https://github.com/jina-ai/finetuner/commit/f91f39f593c145ae6715293d05bcadca2b4e114b)] __-__ add tailor plus tuner integration test (#124) (*Wang Bo*)
 - [[```56c13e59```](https://github.com/jina-ai/finetuner/commit/56c13e590c0df6788a0335dac1845f5ee0b83023)] __-__ add pr labeler (#123) (*Han Xiao*)
 - [[```562c65f5```](https://github.com/jina-ai/finetuner/commit/562c65f53a178ce634dac709bf55bc49def70900)] __-__ __tuner__: add test for overfitting (#109) (*Tadej Svetina*)
 - [[```b448a611```](https://github.com/jina-ai/finetuner/commit/b448a6112472bced9ab4c1b49b8d2ab960136d5d)] __-__ __tailor__: assure weights are preserved after calling to_embedding_model (#106) (*Wang Bo*)
 - [[```47b7a55d```](https://github.com/jina-ai/finetuner/commit/47b7a55da2d2a24e2a9e2e46991fb31d5c5419f5)] __-__ __tailor__:  add test for name is none (#87) (*Wang Bo*)

### 🍹 Other Improvements

 - [[```370e5fba```](https://github.com/jina-ai/finetuner/commit/370e5fba6b2e8d7d1c916d46af2584399faa1926)] __-__ __cd__: add tag and release note script (*Han Xiao*)
 - [[```33b1c90b```](https://github.com/jina-ai/finetuner/commit/33b1c90b7611174efbca45f881ba8b1291689448)] __-__ update readme (*Han Xiao*)
 - [[```0be69a45```](https://github.com/jina-ai/finetuner/commit/0be69a45c0550dd44cf7bae88c70c76b22d6c47f)] __-__ Introduce catalog + ndcg (#120) (*Maximilian Werk*)
 - [[```8bba726e```](https://github.com/jina-ai/finetuner/commit/8bba726e22e94d73126f06c4fbf5a21d054744e7)] __-__ update svg (*Han Xiao*)
 - [[```dfc334f7```](https://github.com/jina-ai/finetuner/commit/dfc334f7703721f765b5d675630e0b8404bd0b41)] __-__ fix emoji (*Han Xiao*)
 - [[```a589a016```](https://github.com/jina-ai/finetuner/commit/a589a0163492e749546e0e69a57226e043bd2db8)] __-__ __docs__: add note from get_framework (*Han Xiao*)
 - [[```d970a2b6```](https://github.com/jina-ai/finetuner/commit/d970a2b63e0712483f119d27cda7a966019904fb)] __-__ fix styling (*Han Xiao*)
 - [[```62a0da7e```](https://github.com/jina-ai/finetuner/commit/62a0da7e2b0cf938dd63a6bbd28d32d693c1f486)] __-__ __version__: the next version will be 0.0.3 (*Jina Dev Bot*)

<a name=release-note-0-0-4></a>
## Release Note (`0.0.4`)

> Release time: 2021-10-20 08:53:48



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Jina Dev Bot,  🙇


### 📗 Documentation

 - [[```6854ba0b```](https://github.com/jina-ai/finetuner/commit/6854ba0ba8ca7b9fa08c5fc12f55734cc304da03)] __-__ fix ecosystem sidebar (*Han Xiao*)

### 🍹 Other Improvements

 - [[```0007fd84```](https://github.com/jina-ai/finetuner/commit/0007fd84cde8b4b59f6f50ef2a6e6325af2e4cc6)] __-__ fix logos (*Han Xiao*)
 - [[```400e8070```](https://github.com/jina-ai/finetuner/commit/400e8070ba56f7a4b4b566cf94031a1615901ca1)] __-__ update readme (*Han Xiao*)
 - [[```73421284```](https://github.com/jina-ai/finetuner/commit/734212848d3276d29a7bb4ba9e7da59db77aef13)] __-__ fix setup.py (*Han Xiao*)
 - [[```db3757d4```](https://github.com/jina-ai/finetuner/commit/db3757d49694d70925f9158bd5be9a18d095a5f2)] __-__ fix readme (*Han Xiao*)
 - [[```1a3002b6```](https://github.com/jina-ai/finetuner/commit/1a3002b66419515b817566713413ba6527b01c79)] __-__ __version__: the next version will be 0.0.4 (*Jina Dev Bot*)

<a name=release-note-0-1-0></a>
## Release Note (`0.1.0`)

> Release time: 2021-10-20 09:04:47



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Jina Dev Bot,  🙇


### 🐞 Bug fixes

 - [[```f6ba40d0```](https://github.com/jina-ai/finetuner/commit/f6ba40d0c9619a92c19a0a19d2ba6a71a419ea8b)] __-__ __setup__: add MANIFEST.in (*Han Xiao*)

### 🍹 Other Improvements

 - [[```377959a1```](https://github.com/jina-ai/finetuner/commit/377959a14976d6f3ec99165bcbd23bf6132e2477)] __-__ __version__: the next version will be 0.0.5 (*Jina Dev Bot*)

<a name=release-note-0-1-1></a>
## Release Note (`0.1.1`)

> Release time: 2021-10-24 11:03:40



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Wang Bo,  Deepankar Mahapatro,  Mohammad Kalim Akram,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```43480cc3```](https://github.com/jina-ai/finetuner/commit/43480cc312bc2134d88a7fe4105766877b49cbcd)] __-__ __helper__: set_embedding function for all frameworks (#163) (*Han Xiao*)
 - [[```fddc57dc```](https://github.com/jina-ai/finetuner/commit/fddc57dc1200abb4c0eeb889778300bf5f82c23e)] __-__ __labeler__: allow user fixing the question (#159) (*Han Xiao*)

### 🐞 Bug fixes

 - [[```1e07e34c```](https://github.com/jina-ai/finetuner/commit/1e07e34c60b3455921f3e66329e276c4b1557889)] __-__ reset toggle on reload (#154) (*Mohammad Kalim Akram*)

### 🧼 Code Refactoring

 - [[```d8d875ff```](https://github.com/jina-ai/finetuner/commit/d8d875ff86745e2bafd83df99ff18d885c85c21a)] __-__ __labeler__: use set_embeddings in labeler (#165) (*Han Xiao*)

### 📗 Documentation

 - [[```d1a9396d```](https://github.com/jina-ai/finetuner/commit/d1a9396df03685710eb61a67954784f22ed0db69)] __-__ remind user again change the data pth (#158) (*Wang Bo*)
 - [[```b92df7de```](https://github.com/jina-ai/finetuner/commit/b92df7de90cf07c345971dc1af354e70bdee9708)] __-__ enable docbot for finetuner (#153) (*Deepankar Mahapatro*)

### 🏁 Unit Test and CICD

 - [[```0d8e0b58```](https://github.com/jina-ai/finetuner/commit/0d8e0b587587326ba40bb79d825cc8030161cbda)] __-__ add gpu test for set embedding (#164) (*Wang Bo*)

### 🍹 Other Improvements

 - [[```87cdc133```](https://github.com/jina-ai/finetuner/commit/87cdc1337b41d3bc81bff12b6503511248aec1e6)] __-__ fix docs css styling (*Han Xiao*)
 - [[```8e3b1fbb```](https://github.com/jina-ai/finetuner/commit/8e3b1fbbb3854ecac5c2a9afb339ce338ea34f50)] __-__ fix styling (*Han Xiao*)
 - [[```870c5a23```](https://github.com/jina-ai/finetuner/commit/870c5a233060b2808d9d16a5134c480d496380c0)] __-__ fill missing docstrings (#162) (*Wang Bo*)
 - [[```67896b97```](https://github.com/jina-ai/finetuner/commit/67896b97146b567c4a0c7505f62d0626bfd8adde)] __-__ fix readme (*Han Xiao*)
 - [[```838ebe35```](https://github.com/jina-ai/finetuner/commit/838ebe358e945960b2e1fe663a7c4a63e38295eb)] __-__ update readme (*Han Xiao*)
 - [[```ccf6de1a```](https://github.com/jina-ai/finetuner/commit/ccf6de1a10ff5228a159326f30567331339a7e6e)] __-__ __docs__: fix docs banner (*Han Xiao*)
 - [[```9e4af657```](https://github.com/jina-ai/finetuner/commit/9e4af657aeba05da64c79addfc19b17a6143264a)] __-__ __version__: the next version will be 0.1.1 (*Jina Dev Bot*)

<a name=release-note-0-1-2></a>
## Release Note (`0.1.2`)

> Release time: 2021-10-26 19:03:12



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```df192645```](https://github.com/jina-ai/finetuner/commit/df1926453880976a85eafe5ce0a94f1e81168fcb)] __-__ __labeler__: gently terminate the labler UI from frontend (#177) (*Han Xiao*)
 - [[```115a0aa4```](https://github.com/jina-ai/finetuner/commit/115a0aa45ce913b95b8e93ceb96aa43905092b83)] __-__ __tuner__: add plot function for tuner.summary (#167) (*Han Xiao*)

### 🐞 Bug fixes

 - [[```40261d47```](https://github.com/jina-ai/finetuner/commit/40261d478f67185a9743ae0136010b7895f73fa9)] __-__ __api__: levelup save and display to top-level (#176) (*Han Xiao*)
 - [[```320ec5df```](https://github.com/jina-ai/finetuner/commit/320ec5df11d104fbe36ba7e6d467b159a4fbb1c9)] __-__ __api__: return model and summary in highlevel fit (#175) (*Han Xiao*)

### 🍹 Other Improvements

 - [[```ebb9c8d5```](https://github.com/jina-ai/finetuner/commit/ebb9c8d57b9df1b65f0ce650b72b5ec4446a2a35)] __-__ __setup__: update jina minimum requirement for new block() (*Han Xiao*)
 - [[```1c5d00cd```](https://github.com/jina-ai/finetuner/commit/1c5d00cd0b7aab01fa36ada730614d6d3c410d63)] __-__ __version__: the next version will be 0.1.2 (*Jina Dev Bot*)

<a name=release-note-0-1-3></a>
## Release Note (`0.1.3`)

> Release time: 2021-10-27 07:27:34



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Jina Dev Bot,  🙇


### 🧼 Code Refactoring

 - [[```1ae201a0```](https://github.com/jina-ai/finetuner/commit/1ae201a087ebcf7c80d5c1dbe736a64e0c11a341)] __-__ __embedding__: level up embed method to top API add docs (#178) (*Han Xiao*)

### 🍹 Other Improvements

 - [[```bf07ab12```](https://github.com/jina-ai/finetuner/commit/bf07ab122c23a987722590817bad85c83d108a51)] __-__ __version__: the next version will be 0.1.3 (*Jina Dev Bot*)

<a name=release-note-0-1-4></a>
## Release Note (`0.1.4`)

> Release time: 2021-11-02 21:06:01



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Wang Bo,  Aziz Belaweid,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```1e4a1aee```](https://github.com/jina-ai/finetuner/commit/1e4a1aeebce9c11ec3372a716a1f17c31396b6b8)] __-__ __tuner__: add miner v1 (#180) (*Wang Bo*)
 - [[```ae8e3990```](https://github.com/jina-ai/finetuner/commit/ae8e3990080681a760f465b29c381ffe0e4b0245)] __-__ __helper__: add batch_size to embed fn (#183) (*Han Xiao*)

### 📗 Documentation

 - [[```d21345a3```](https://github.com/jina-ai/finetuner/commit/d21345a3201ec6c9e920a41bfe59cf53e6a0524e)] __-__ update according to new jina api (*Han Xiao*)
 - [[```7e9c04fa```](https://github.com/jina-ai/finetuner/commit/7e9c04faebc649a45e032d4ef86040b3342824d5)] __-__ added resize to fix keras shape error (#174) (*Aziz Belaweid*)

### 🍹 Other Improvements

 - [[```1ce3d8e1```](https://github.com/jina-ai/finetuner/commit/1ce3d8e1e4e343968083a9e54b2e31b61160c544)] __-__ bump jina requirements (*Han Xiao*)
 - [[```43d62f06```](https://github.com/jina-ai/finetuner/commit/43d62f068fdf80ca0a1a4d9f86ec24804f7f6aca)] __-__ __readme__: update logo (*Han Xiao*)
 - [[```489014ee```](https://github.com/jina-ai/finetuner/commit/489014ee4e12e6c3cc697d7c4da4129dd600ccdb)] __-__ __version__: the next version will be 0.1.4 (*Jina Dev Bot*)

<a name=release-note-0-1-5></a>
## Release Note (`0.1.5`)

> Release time: 2021-11-08 10:20:47



🙇 We'd like to thank all contributors for this new release! In particular,
 Roshan Jossy,  Han Xiao,  Wang Bo,  Tadej Svetina,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```531d9052```](https://github.com/jina-ai/finetuner/commit/531d9052c5af93327974aa31a1da47f14bdc3239)] __-__ __tuner__: add miner for session dataset (#184) (*Wang Bo*)
 - [[```77df7676```](https://github.com/jina-ai/finetuner/commit/77df7676f2fb39cda98fb7d87be8b5e21c49c421)] __-__ reformat data loading (#181) (*Tadej Svetina*)

### 🐞 Bug fixes

 - [[```3d5fc769```](https://github.com/jina-ai/finetuner/commit/3d5fc7691ee7591b489e1db37fbc22fabb75680e)] __-__ __embedding__: fix embedding train/eval time behavior (#190) (*Han Xiao*)

### 🏁 Unit Test and CICD

 - [[```d80a4d0f```](https://github.com/jina-ai/finetuner/commit/d80a4d0faba7b504dcc93fe9769242f331df1484)] __-__ __embedding__: add test for #190 (*Han Xiao*)
 - [[```fd1fe384```](https://github.com/jina-ai/finetuner/commit/fd1fe3843d5d4130065100194694b57216a71953)] __-__ upgrade tf version (#189) (*Wang Bo*)
 - [[```5059e202```](https://github.com/jina-ai/finetuner/commit/5059e20280b9c5c8008577f63fd5404c08b40db1)] __-__ pin framework version (#188) (*Wang Bo*)

### 🍹 Other Improvements

 - [[```e1a73434```](https://github.com/jina-ai/finetuner/commit/e1a734346a69c8763072fa43b536224363cb8509)] __-__ __labeler__: add component for audio matches (#185) (*Roshan Jossy*)
 - [[```717f06a0```](https://github.com/jina-ai/finetuner/commit/717f06a06ecb97eebc54b4448e865dbd4adc24e0)] __-__ __version__: the next version will be 0.1.5 (*Jina Dev Bot*)

<a name=release-note-0-2-0></a>
## Release Note (`0.2.0`)

> Release time: 2021-11-19 14:22:57



🙇 We'd like to thank all contributors for this new release! In particular,
 Han Xiao,  Yanlong Wang,  Tadej Svetina,  Wang Bo,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```f920fe25```](https://github.com/jina-ai/finetuner/commit/f920fe257d2a709e9ff9015188040cce242b5cdf)] __-__ reformat pipeline (#192) (*Tadej Svetina*)

### 🐞 Bug fixes

 - [[```fe67bb92```](https://github.com/jina-ai/finetuner/commit/fe67bb92610f40c6b6ec2c1949338c806c6b2277)] __-__ docs celeba (#211) (*Tadej Svetina*)
 - [[```e0f81474```](https://github.com/jina-ai/finetuner/commit/e0f814746dd0352febc698171903c6e3d59cc51c)] __-__ make get_framework robust (#207) (*Tadej Svetina*)
 - [[```376f4028```](https://github.com/jina-ai/finetuner/commit/376f4028e9df679e5fa573b045c152bc4333d0e7)] __-__ __tailor__: fix to emebdding model (#196) (*Wang Bo*)

### 📗 Documentation

 - [[```0a67481d```](https://github.com/jina-ai/finetuner/commit/0a67481d8a043019ec74a8a1a8267b32de4c1473)] __-__ fix doc-bot style during load (#212) (*Yanlong Wang*)

### 🍹 Other Improvements

 - [[```c6041fde```](https://github.com/jina-ai/finetuner/commit/c6041fde2bddf74d50922a3a55c3bff14ff17f46)] __-__ __version__: set next version to 0.2.0 (*Han Xiao*)
 - [[```20eb41c2```](https://github.com/jina-ai/finetuner/commit/20eb41c2e0504d2781ca3fd7d915b97aa0118129)] __-__ __style__: fix coding style optimize imports (*Han Xiao*)
 - [[```6539237b```](https://github.com/jina-ai/finetuner/commit/6539237befd25205707d134dbd4aa3897b13ea3d)] __-__ __version__: the next version will be 0.1.6 (*Jina Dev Bot*)

