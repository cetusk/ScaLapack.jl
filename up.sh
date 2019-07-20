#!/bin/sh

scalapackpkg=~/.julia/packages/ScaLapack/J8GM7
localscalapackpkg=./

cp -r $localscalapackpkg/* $scalapackpkg/

rm $scalapackpkg/up.sh $scalapackpkg/.DS_Store
