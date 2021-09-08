const urlParams = new URLSearchParams(window.location.search);
const app = new Vue({
    el: '#app',
    data: {
        is_busy: false,
        is_conn_broken: false,
        view_template: {
            content: [{text: '.uri', value: 'uri'},
                {text: '.text', value: 'text'},
                {text: '.tags', value: 'tags'}],
            style: [{text: 'List', value: 'list'},
                {text: 'Pair', value: 'pair'}],
        },
        labeler_config: {
            content: 'uri',
            style: 'list',
            example_per_view: 3,
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
        },
        general_config: {
            server_port: 65123,
            server_address: `http://localhost`,
            next_endpoint: '/next',
            fit_endpoint: '/fit'
        },
        advanced_config: {
            pos_value: {text: 'Positive label', value: 1},
            neg_value: {text: 'Negative label', value: -1},
            epochs: {text: 'Epochs', value: 10},
            sample_size: {text: 'Match pool', value: 1000}
        },
        cur_batch: []
    },
    computed: {
        host_address: function () {
            return `${this.general_config.server_address}:${urlParams.get('port') ?? this.general_config.server_port}`
        },
        next_address: function () {
            return `${this.host_address}${this.general_config.next_endpoint}`
        },
        fit_address: function () {
            return `${this.host_address}${this.general_config.fit_endpoint}`
        },
        positive_rate: function () {
            return this.progress_stats.positive.value / (this.progress_stats.positive.value + this.progress_stats.negative.value) * 100
        },
        negative_rate: function () {
            return this.progress_stats.negative.value / (this.progress_stats.positive.value + this.progress_stats.negative.value) * 100
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
                    match.tags.finetuner.label = app.advanced_config.pos_value.value
                    app.progress_stats.positive.value++
                } else {
                    match.tags.finetuner.label = app.advanced_config.neg_value.value
                    app.progress_stats.negative.value++
                }
                delete match.tags.finetuner_label
            });
            app.progress_stats.total.value++
            app.fit_doc(doc)
            app.next_batch()
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
        next_batch: function () {
            let end_idx = app.labeler_config.start_idx + (app.labeler_config.example_per_view - app.cur_batch.length)
            if (end_idx === app.labeler_config.start_idx) {
                return
            }
            let start_idx = app.labeler_config.start_idx
            app.labeler_config.start_idx = end_idx
            app.is_busy = true
            app.is_conn_broken = false
            $.ajax({
                type: "POST",
                url: app.next_address,
                data: JSON.stringify({
                    data: [],
                    parameters: {
                        'start': start_idx,
                        'end': end_idx,
                        'topk': app.labeler_config.topk_per_example,
                        'sample_size': app.advanced_config.sample_size.value
                    }
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
            }).success(function (data, textStatus, jqXHR) {
                app.cur_batch.push(...data['data'].docs)
                app.is_busy = false
                app.progress_stats.this_session.value = app.cur_batch.length
            }).fail(function () {
                console.error("bad connection!")
                app.is_conn_broken = true
                app.is_busy = false
            });
        }
    }
});


Vue.nextTick(function () {
    app.next_batch()
})