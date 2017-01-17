<url-input>
  <div class="row form-group url-input-group form-baseline">
    <label class="col-4 col-md-2 form-control-label text-right">
      <button
        if={ !opts.first }
        onclick={ opts.onremove }
        type="button"
        class="remove-url btn btn-outline-danger btn-sm"
        tabindex="-1"
      >
        <i class="fa fa-times-circle"></i>
      </button>
      <span class="text">URL { opts.index }</span>
    </label>

    <div class="col-4 col-md-8">
      <input
        type="url"
        class="form-control"
        name="urls[]"
        placeholder="URL"
        value={ opts.url.value }
        >
    </div>

    <div class="col-4 col-md-2 add-url-group">
      <button
        if={ opts.last }
        onclick={ opts.onadd }
        type="button"
        class="btn btn-secondary add-url btn-sm"
      >Add URL</button>
    </div>
  </div>
</url-input>
