'use strict';

const gulp = require('gulp');
const livereload = require('gulp-livereload');

const files = ['__init__.py', 'static/**', 'templates/**'];

gulp.task('livereload', function() {
  gulp.src(files).pipe(livereload());
});

gulp.task('watch', function() {
  livereload.listen();
  gulp.watch(files, ['livereload']);
});
