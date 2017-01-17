'use strict';

const path = require('path');
const webpack = require('webpack');
const ExtractTextPlugin = require('extract-text-webpack-plugin');
const ManifestPlugin = require('webpack-manifest-plugin');

const suffix = process.env.NODE_ENV === 'production' ? '[hash].' : '';

module.exports = {
  context: __dirname,
  entry: {
    app: './frontend'
  },
  output: {
    path: path.join(__dirname, 'static'),
    filename: `app.${suffix}js`
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
  plugins: [
    new ExtractTextPlugin(`app.${suffix}css`),
    new ManifestPlugin()
  ]
};
