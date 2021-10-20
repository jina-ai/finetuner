



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

