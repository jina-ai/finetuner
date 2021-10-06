const textMatchCard = {
    props: {
      doc: Object,
      docIdx: Number,
      getContent: Function,
      toggleRelevance: Function,
      invertSelection: Function,
      submitDoc: Function,
    },
    template: `
        <div class="card text-card">
            <div class="card-header">
                <p class="fs-6 fw-light mb-2">Select all texts similar to:</p>
                <h3>{{getContent(doc)}}</h3>
            </div>
            <div class="card-body">
                <ul class="list-group">
                    <li class="list-group-item" v-for="(match, matchIndex) in doc.matches">
                        <div class="match-item" :class="{ 'positive-match': match.tags.finetuner_label }"
                                style="width: 100%" v-on:click="toggleRelevance(match)">
                            {{getContent(match)}}
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