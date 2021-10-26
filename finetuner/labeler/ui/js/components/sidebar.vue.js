const sidebar = {
    props: {
        labelerConfig: Object,
        viewTemplate: Object,
        tags: Array,
        isBusy: Boolean,
        progressStats: Object,
        positiveRate: Number,
        negativeRate: Number,
        advancedConfig: Object,
        saveProgress: Function,
        nextBatch: Function,
        terminateFlow: Function,
    },
    template: `
    <div class="d-flex flex-column flex-shrink-0 p-3 sidebar">
        <a href="/" class="d-flex align-items-center mb-3 mx-md-auto text-decoration-none">
                <span class="app-title">
                    <img src="img/logo-light.svg" width="40%"
                         alt="Finetuner logo: Finetuner allows one to finetune any deep Neural Network for better embedding on search tasks.">
                </span>
        </a>

        <div class="accordion" id="accordionPanelsStayOpenExample">
            <div class="accordion-item">
                <h2 class="accordion-header" id="panelsStayOpen-headingOne">
                    <button class="accordion-button options-title" type="button" data-bs-toggle="collapse"
                            data-bs-target="#panelsStayOpen-collapseOne" aria-expanded="true"
                            aria-controls="panelsStayOpen-collapseOne">
                        View
                    </button>
                </h2>
                <div id="panelsStayOpen-collapseOne" class="accordion-collapse collapse show"
                    aria-labelledby="panelsStayOpen-headingOne">
                    <div class="accordion-body">
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">Field</label>
                            <div class="col-sm-6">
                                <select class="form-select" v-model="labelerConfig.content">
                                    <option v-for="option in viewTemplate.content" v-bind:value="option.value">
                                        {{ option.text }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="row my-1" v-if="labelerConfig.content == 'tags'">
                            <label class="col-sm-6 col-form-label">Tags Key</label>
                            <div class="col-sm-6">
                                <select class="form-select" v-model="labelerConfig.tags">
                                    <option v-for="option in tags" v-bind:value="option">
                                        {{ option }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">Content Type</label>
                            <div class="col-sm-6">
                                <select class="form-select" v-model="labelerConfig.style">
                                    <option v-for="option in viewTemplate.style" v-bind:value="option.value">
                                        {{ option.text }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">Start question</label>
                            <div class="col-sm-6">
                                <input class="form-control" type="number" min="0"
                                        v-model.number="labelerConfig.start_idx" v-on:input="nextBatch()" 
                                        :disabled="labelerConfig.same_question">
                            </div>
                        </div>
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">Questions/Session</label>
                            <div class="col-sm-6">
                                <input class="form-control" type="number" min="1" max="9"
                                        v-model.number="labelerConfig.example_per_view" v-on:input="nextBatch()"
                                        :disabled="labelerConfig.same_question">
                            </div>
                        </div>
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">Keep same question</label>
                            <div class="col-sm-6 d-flex align-items-center justify-content-center">
                                <input class="form-check-input"  type="checkbox"
                                        v-model="labelerConfig.same_question" v-on:input="nextBatch(true, false)">
                            </div>
                        </div>
                        <div class="row my-1">
                            <label class="col-sm-6 col-form-label">TopK/Question</label>
                            <div class="col-sm-6">
                                <input class="form-control" type="number" min="1" max="9"
                                        v-model.number="labelerConfig.topk_per_example" v-on:input="nextBatch()">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="accordion-item">
                <h2 class="accordion-header" id="panelsStayOpen-headingTwo">
                    <button class="accordion-button options-title" type="button" data-bs-toggle="collapse"
                            data-bs-target="#panelsStayOpen-collapseTwo" aria-expanded="true"
                            aria-controls="panelsStayOpen-collapseTwo">
                        Progress
                        <div class="mx-2 spinner-border spinner-border-sm align-middle" role="status"
                                v-show="isBusy">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </button>
                </h2>
                <div id="panelsStayOpen-collapseTwo" class="accordion-collapse collapse show"
                        aria-labelledby="panelsStayOpen-headingTwo">
                    <div class="accordion-body">
                        <div class="row my-1" v-for="option in progressStats">
                            <label class="col-sm-6 col-form-label">{{ option.text }}</label>
                            <div class="col-sm-6 text-end">
                                {{ option.value }}
                            </div>
                        </div>
                        <div class="progress">
                            <div class="progress-bar progress-positive" role="progressbar"
                                    :style="{width: positiveRate+'%' }"></div>
                            <div class="progress-bar progress-negative" role="progressbar"
                                    :style="{width: negativeRate+'%' }"></div>
                        </div>
                    </div>
                    <div class="my-3 d-flex justify-content-center">
                      <button class="btn btn btn-outline-primary m-2"
                          v-on:click="saveProgress()">
                          Save model
                      </button>
                      <button class="btn btn btn-outline-secondary m-2"
                          v-on:click="terminateFlow()">
                          Terminate
                      </button>
                    </div>
                </div>
            </div>
            <div class="accordion-item">
                <h2 class="accordion-header" id="panelsStayOpen-headingThree">
                    <button class="accordion-button options-title collapsed" type="button" data-bs-toggle="collapse"
                            data-bs-target="#panelsStayOpen-collapseThree" aria-expanded="false"
                            aria-controls="panelsStayOpen-collapseThree">
                        Advanced
                    </button>
                </h2>
                <div id="panelsStayOpen-collapseThree" class="accordion-collapse collapse"
                        aria-labelledby="panelsStayOpen-headingThree">
                    <div class="accordion-body">
                        <div class="row my-1" v-for="option in advancedConfig">
                            <label class="col-sm-6 col-form-label">{{ option.text }}</label>
                            <div class="col-sm-6 text-end">
                                <input class="form-control" :type="option.type" v-model.number="option.value">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <a href="https://jina.ai" target="_blank">© 2020-2021 Jina AI</a>
            <a href="https://github.com/jina-ai/finetuner" target="_blank">(<strong>Finetuner</strong> v0.1)</a>
        </footer>
    </div>
  `,
    mounted() {
        console.log("sidebar is loaded")
    }
}