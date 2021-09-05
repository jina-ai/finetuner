var app = new Vue({
    el: '#app',
    data: {
        view_template: {
            content: ['uri', 'text', 'tags'],
            style: ['List', 'Pair'],
        },
        labeler_config: {
            content: 'uri',
            style: 'List',
            example_per_view: 1,
        }
    }
})