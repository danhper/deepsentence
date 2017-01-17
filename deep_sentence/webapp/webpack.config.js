'use strict';

const path = require('path');
const webpack = require('webpack');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

module.exports = {
  context: __dirname,
  entry: './frontend',
  output: {
    path: path.join(__dirname, 'static'),
    filename: 'app.js'
  },
  module: {
    preLoaders: [
      { test: /\.tag$/, exclude: /node_modules/, loader: 'riotjs', query: { type: 'none' } }
    ],
    loaders: [
      { test: /\.js$|\.tag$/, exclude: /node_modules/, loader: 'babel?presets[]=es2015' },
      { test: /\.less$/, loader: ExtractTextPlugin.extract('style-loader', 'css!less') }
    ]
  },
  plugins: [new ExtractTextPlugin('app.css')]
};
