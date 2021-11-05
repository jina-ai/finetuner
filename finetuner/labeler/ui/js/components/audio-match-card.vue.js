const audioMatchCard = {
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
            <div class="card-header flex-column">
                <p class="fs-6 fw-light mb-2 p-2">Select all sounds similar to the following one</p>
                <audio
                    v-bind:src="getContent(doc)" 
                    v-on:click="toggleRelevance(match)"
                    class=""
                    controls
                    />
                </model-viewer>
            </div>
            <div class="card-body">
                <ul class="list-group">
                    <li class="list-group-item" v-for="(match, matchIndex) in doc.matches">
                        <div class="match-item" :class="{ 'positive-match': match.tags.finetuner_label }"
                            style="width: 100%" v-on:click="toggleRelevance(match)">
                            <audio
                                v-bind:src="getContent(match)" 
                                v-on:click="toggleRelevance(match)"
                                class=""
                                controls
                                />
                            <div class="kbd">{{matchIndex}}</div>
                        </div>

                    </li>
                </ul>
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