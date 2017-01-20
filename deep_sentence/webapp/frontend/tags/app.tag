require('./raw-html.tag');
require('./checklist-item.tag');

import route from 'riot-route';
import queryString from 'query-string';
import axios from 'axios';

<app>
  <h1 class="text-center">Deep Sentence</h1>

  <div if={ error } class="row text-center">
    <div class="col-sm-12 text-center">
      <div class="alert alert-danger alert-dismissible alert-auto" role="alert">
        <button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>
        An error occured: { error }
      </div>
    </div>
  </div>

  <p>Enter the URLs you want to summarize</p>
  <form onsubmit={ submitForm }>
    <div class="urls">
      <url-inputs urls={ urls } />
    </div>

    <div class="form-group row">
      <div class="col-sm-12 text-center">
        <button type="submit" class={ disabled: loading, btn: true, 'btn-primary': true } disabled={ loading }>
          <i if={ !loading } class="fa fa-list"></i>
          <i if={ loading } class="fa fa-spinner fa-spin fa-fw"></i>
          Summarize
        </button>
      </div>
    </div>
  </form>

  <div if={ loading } class="text-center">
    <ul class="list-group checklist">
      <checklist-item item-id={ 0 } progress={ progress } text="Fetch articles" />
      <checklist-item item-id={ 1 } progress={ progress } text="Summarize articles" />
      <checklist-item item-id={ 2 } progress={ progress } text="Generate title" />
    </ul>
  </div>

  <div if={ summary } class="summary-result">
    <hr class="separator">
    <div class="result">
      <h2 class="text-center">{ title }</h2>
      <p class="summary"><raw-html content={ summary } /></p>
    </div>
  </div>

  <script>
    route.base('/');

    const clientId = Math.random().toString(36).substring(7);
    axios.defaults.headers.common['x-client-id'] = clientId;

    const socket = io.connect({transports: ['polling']});
    socket.on('connect', () => {
      socket.emit('subscribe', {room: clientId});
    });

    this.progress = -1;

    socket.on('progress', message => {
      this.progress = message.progress;
      this.update();
    });

    _.assign(this, _.pick(opts, ['summary', 'title', 'error']));

    this.processQueryUrls = () => {
      let queryUrls = queryString.parse(location.search)['urls[]'];
      if (!queryUrls) {
        this.urls = [{ value: '' }];
        return;
      }
      if (!(queryUrls instanceof Array)) {
        queryUrls = [queryUrls];
      }
      this.fetchSummary(queryUrls);
    }

    const makeURLQuery = (urls) => {
      return _.map(urls, url => `urls[]=${encodeURIComponent(url)}`).join('&');
    };

    this.submitForm = (e) => {
      e.preventDefault();
      const urls = _(e.target.elements)
                   .filter(elem => elem.name === 'urls[]')
                   .map(elem => elem.value)
                   .value();
      if (_.isEqual(urls, this.urls)) {
        return;
      }
      this.fetchSummary(urls);
    };

    this.fetchSummary = (urls) => {
      if (this.loading) {
        return;
      }

      this.urls = _.map(urls, url => { return { value: url } });
      const query = makeURLQuery(urls);
      route(`/?${query}`);

      this.summary = this.error = this.title =  '';
      this.loading = true;
      this.progress = 0;
      axios.get(`/summary.json?${query}`).then(res => {
        this.update({title: res.data.title, summary: res.data.summary, loading: false});
        this.progress = 4;
      }).catch(e => {
        this.update({error: e.response.data.error, loading: false, progress: -1});
      });
    };

    this.processQueryUrls();
  </script>
</app>
