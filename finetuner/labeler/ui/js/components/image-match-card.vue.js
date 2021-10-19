const imageMatchCard = {
    props: {
      doc: Object,
      docIdx: Number,
      getContent: Function,
      toggleRelevance: Function,
      invertSelection: Function,
      submitDoc: Function,
    },
    template: `
        <div class="card image-card">
            <div class="card-header">
                <p class="fs-6 fw-light mb-2 hint-text">Select all images similar to the image on right</p>
                <img v-bind:src="getContent(doc)" class="img-thumbnail img-fluid my-2">
            </div>
            <div class="card-body">
                <div class="image-matches-container">
                    <div class="col compact-img" v-for="(match, matchIndex) in doc.matches">
                        <div class="d-flex justify-content-center" v-bind:class="{ 'positive-match': match.tags.finetuner_label }">
                            <img v-bind:src="getContent(match)" class="img-thumbnail img-fluid"
                                    v-on:click="toggleRelevance(match)">
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