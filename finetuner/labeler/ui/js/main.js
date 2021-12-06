const urlParams = new URLSearchParams(window.location.search);
Vue.use(VueAwesomeSwiper)
const app = new Vue({
    el: '#app',
    components: {
        "sidebar": sidebar,
        "image-match-card": imageMatchCard,
        "text-match-card": textMatchCard,
        "mesh-match-card": meshMatchCard,
        "audio-match-card": audioMatchCard,
    },
    data: {
        is_busy: false,
        is_conn_broken: false,
        view_template: {
            content: [{text: '.uri', value: 'uri'},
                {text: '.text', value: 'text'},
                {text: '.tags', value: 'tags'}],
            style: [{text: 'Image', value: 'image'},
                {text: 'Text', value: 'text'},
                {text: 'Audio', value: 'audio'},
                {text: '3D mesh', value: 'mesh'}],
        },
        labeler_config: {
            content: 'uri',
            style: 'image',
            example_per_view: 3,
            same_question: false,
            topk_per_example: 9,
            tags: '',
            start_idx: 0,
        },
        progress_stats: {
            this_session: {text: 'This Session', value: 0},
            total: {text: 'Done', value: 0},
            positive: {text: 'Positive', value: 0},
            negative: {text: 'Negative', value: 0},
            ignore: {text: 'Ignore', value: 0},
            saved: {text: 'Saved', value: 0}
        },
        general_config: {
            server_port: 65123,
            server_address: `http://localhost`,
            next_endpoint: '/next',
            fit_endpoint: '/fit',
            save_endpoint: '/save',
            stop_endpoint: '/terminate',
        },
        advanced_config: {
            pos_value: {text: 'Positive label', value: 1, type: 'number'},
            neg_value: {text: 'Negative label', value: -1, type: 'number'},
            epochs: {text: 'Epochs', value: 1, type: 'number'},
            sample_size: {text: 'Match pool', value: 1000, type: 'number'},
            model_path: {text: 'Model save path', value: 'tuned-model', type: 'text'}
        },
        cur_batch: [],
        tags: [],
        swiperOptions: {
            effect: 'flip',
            grabCursor: false,
            centeredSlides: true,
            slidesPerView: 3,
            allowTouchMove: false,
            keyboard: {
                enabled: true,
            },
            pagination: {
                el: '.swiper-pagination'
            },
            navigation: {
                nextEl: '.swiper-button-next',
                prevEl: '.swiper-button-prev'
            }
        }
    },
    computed: {
        host_address: function () {
            return `${this.general_config.server_address}:${location.port ?? this.general_config.server_port}`
        },
        next_address: function () {
            return `${this.host_address}${this.general_config.next_endpoint}`
        },
        fit_address: function () {
            return `${this.host_address}${this.general_config.fit_endpoint}`
        },
        save_address: function () {
            return `${this.host_address}${this.general_config.save_endpoint}`
        },
        stop_address: function () {
            return `${this.host_address}${this.general_config.stop_endpoint}`
        },
        positive_rate: function () {
            return this.progress_stats.positive.value / (this.progress_stats.positive.value + this.progress_stats.negative.value) * 100
        },
        negative_rate: function () {
            return this.progress_stats.negative.value / (this.progress_stats.positive.value + this.progress_stats.negative.value) * 100
        }
    },
    watch: {
        'labeler_config.content': (newContent) => {
            if (newContent === 'text' || newContent === 'tags') {
                app.labeler_config.style = 'text'
                app.labeler_config.tags = app.tags.length > 0 ? app.tags[0] : ''
            } else if (newContent === 'uri') app.labeler_config.style = 'image'
        }
    },
    methods: {
        toggle_relevance: function (match) {
            Vue.set(match.tags, 'finetuner_label', !match.tags.finetuner_label)
        },
        select_all: function (matches) {
            matches.forEach(function (doc) {
                Vue.set(doc.tags, 'finetuner_label', !doc.tags.finetuner_label)
            });
        },
        submit_doc: function (doc_idx) {
            let doc = app.cur_batch[doc_idx]
            app.cur_batch.splice(doc_idx, 1)
            doc.matches.forEach(function (match) {
                match.tags.finetuner = {}
                if (match.tags.finetuner_label) {
                    match.tags.finetuner_label = app.advanced_config.pos_value.value
                    app.progress_stats.positive.value++
                } else {
                    match.tags.finetuner_label = app.advanced_config.neg_value.value
                    app.progress_stats.negative.value++
                }
            });
            app.progress_stats.total.value++
            app.fit_doc(doc)
            if (app.labeler_config.same_question) {
                app.next_batch(true, false)
            } else {
                app.next_batch(false)
            }
        },
        fit_doc: function (doc) {
            app.is_busy = true
            app.is_conn_broken = false
            $.ajax({
                type: "POST",
                url: app.fit_address,
                data: JSON.stringify({
                    data: [doc],
                    parameters: {
                        epochs: app.advanced_config.epochs.value
                    }
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
            }).success(function (data, textStatus, jqXHR) {
                app.is_busy = false
            }).fail(function () {
                console.error("bad connection!")
                app.is_conn_broken = true
                app.is_busy = false
            });
        },
        get_content: function (doc) {
            if (app.labeler_config.content === 'uri') {
                return doc.uri
            } else if (app.labeler_config.content === 'text') {
                return doc.text
            } else if (app.labeler_config.content === 'tags') {
                return doc.tags[app.labeler_config.tags]
            }
        },
        next_batch: function (clear_exist=true, update_start_idx=true) {
            if (clear_exist) {
                app.cur_batch = []
            }
            let new_examples_to_query = Math.max(0, app.labeler_config.example_per_view - app.cur_batch.length)
            let end_idx = app.labeler_config.start_idx + new_examples_to_query
            if (end_idx <= app.labeler_config.start_idx) {
                return
            }
            app.is_busy = true
            app.is_conn_broken = false
            $.ajax({
                type: "POST",
                url: app.next_address,
                data: JSON.stringify({
                    data: [],
                    parameters: {
                        'start': app.labeler_config.start_idx,
                        'end': end_idx,
                        'topk': app.labeler_config.topk_per_example,
                        'sample_size': app.advanced_config.sample_size.value
                    }
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
            }).success(function (data, textStatus, jqXHR) {
                if (update_start_idx) {
                    app.labeler_config.start_idx = end_idx
                }

                app.cur_batch.push(...data['data'].docs)
                try {
                    app.tags = Object.keys(data.data.docs[0].tags)
                } catch (e) {}

                app.is_busy = false
                app.progress_stats.this_session.value = app.cur_batch.length
            }).fail(function () {
                console.error("bad connection!")
                app.is_conn_broken = true
                app.is_busy = false
            });
        },
        saveProgress: () => {
            app.is_busy = true
            app.is_conn_broken = false
            $.ajax({
                type: "POST",
                url: app.save_address,
                data: JSON.stringify({
                    data: [],
                    parameters: {
                        'model_path': app.advanced_config.model_path.value
                    }
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
            }).success(function (data, textStatus, jqXHR) {
                app.is_busy = false
                app.progress_stats.saved.value++
            }).fail(function () {
                console.error("Error: ", error)
                app.is_busy = false
            });
        },
        terminateFlow: () => {
            app.is_busy = true
            app.is_conn_broken = false
            $.ajax({
                type: "POST",
                url: app.stop_address,
                data: JSON.stringify({
                    data: [],
                    parameters: {
                    }
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
            }).success(function (data, textStatus, jqXHR) {
                app.is_busy = false
                close();
            }).fail(function () {
                console.error("Error: ", error)
                app.is_busy = false
            });
        },
        handleKeyPress(event) {
            let key = event.key
            if (event.target instanceof HTMLInputElement) {
                return
            }
            let {activeIndex} = this.$refs.swiperComponent.$swiper
            let currentDoc = this.cur_batch[activeIndex]
            if (/\d/.test(key)) {
                app.toggle_relevance(currentDoc.matches[parseInt(key, 10)])
            } else if (key === ' ') {
                app.submit_doc(activeIndex)
            } else if (key === 'i') {
                app.select_all(currentDoc.matches)
            }
        },
        onSetTranslate() {
        },
        onSwiperSlideChangeTransitionStart() {
        },
        onSwiperClickSlide(index, reallyIndex) {
        }
    },
    created() {
        window.addEventListener("keydown", this.handleKeyPress);
    },
    destroyed() {
        window.removeEventListener("keydown", this.handleKeyPress);
    },
});


Vue.nextTick(function () {
    app.next_batch()
})
