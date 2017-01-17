require('./url-input.tag');

import queryString from 'query-string';

<url-inputs>
  <url-input
    each={ url, i in urls }
    url={ url }
    onadd={ parent.addURL }
    onremove={ parent.removeURL }
    index={ i + 1 }
    first={ i === 0 }
    last={ i === parent.urls.length - 1 } />

  <script>
    this.urls = [{ value: '' }];

    let queryUrls = queryString.parse(location.search)['urls[]'];
    if (queryUrls) {
      if (!(queryUrls instanceof Array)) {
        queryUrls = [queryUrls];
      }
      this.urls = queryUrls.map(url => { return { value: url } });
    }

    this.addURL = () => {
      this.urls.push({value: ''});
      this.update();
    };

    this.removeURL = (e) => {
      this.urls.splice(e.item.i, 1);
      this.update();
    }
  </script>
</url-inputs>
