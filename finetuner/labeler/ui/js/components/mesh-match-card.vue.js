const meshMatchCard = {
    props: {
      doc: Object,
      docIdx: Number,
      getContent: Function,
      toggleRelevance: Function,
      invertSelection: Function,
      submitDoc: Function,
    },
    template: `
        <div class="card mesh-card">
            <div class="card-header">
                <p class="fs-6 fw-light mb-2 hint-text">Select all meshes similar to the image on right</p>
                <model-viewer
                    v-bind:src="getContent(doc)" 
                    v-on:click="toggleRelevance(match)"
                    class="img-thumbnail img-fluid my-2 h-100"
                    alt="result mesh"
                    ar ar-modes="webxr scene-viewer quick-look"
                    environment-image="neutral"
                    interaction-policy="allow-when-focused"
                    interaction-prompt="when-focused"
                    auto-rotate
                    rotation-per-second="30deg"
                    orientation="0 0 180deg"
                    turntableRotation
                    camera-controls>
                </model-viewer>
            </div>
            <div class="card-body">
                <div class="image-matches-container">
                    <div class="col compact-img" v-for="(match, matchIndex) in doc.matches">
                        <div class="w-100" v-bind:class="{ 'positive-match': match.tags.finetuner_label }">
                            <model-viewer
                                v-bind:src="getContent(match)" 
                                v-on:click="toggleRelevance(match)"
                                class="img-thumbnail img-fluid h-100"
                                alt="result mesh"
                                ar ar-modes="webxr scene-viewer quick-look"
                                environment-image="neutral"
                                interaction-policy="allow-when-focused"
                                interaction-prompt="when-focused"
                                auto-rotate
                                rotation-per-second="30deg"
                                orientation="0 0 180deg"
                                turntableRotation
                                camera-controls>
                            </model-viewer>
                        </div>
                        <div class="kbd">{{matchIndex}}</div>
                    </div>
                </div>
            </div>
            <div class="card-footer">
                <div class="btn-toolbar justify-content-between g-2">
                    <button type="button" class="btn btn-outline-success"
                            v-on:click="invertSelection(doc.matches)">
                        Invert
                        <div class="kbd">i</div>
                    </button>
                    <button type="button" class="btn btn-outline-primary" v-on:click="submitDoc(docIdx)">
                        Done
                        <div class="kbd">space</div>
                    </button>
                </div>
            </div>
        </div>
    `,
    mounted() {
    }
}