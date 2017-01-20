<checklist-item>
  <li class={ 'list-group-item': true, 'item-done': isDone() }>
    <i class={ fa: true,
              'fa-fw': true,
              'fa-square-o': isPending(),
              'fa-check-square': isDone(),
              'fa-spinner': isProgressing(),
              'fa-spin': isProgressing() }></i>
    <span>{ opts.text }</span>
  </li>

  <script>
    this.isPending = () => opts.progress < opts.itemId;
    this.isDone = () => opts.progress > opts.itemId;
    this.isProgressing = () => opts.progress === opts.itemId;
  </script>
</checklist-item>
