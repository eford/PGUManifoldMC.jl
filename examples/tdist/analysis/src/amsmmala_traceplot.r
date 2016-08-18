library(data.table)
library(stringr)

DATADIR <- "../../data"
SUBDATADIR <- "amsmmala"
OUTDIR <- "../output"

nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 10000
ci <- 3
pi <- 7

chains <- t(fread(
  file.path(DATADIR, SUBDATADIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
))

chainmean = mean(chains[, pi])

pdf(file=file.path(OUTDIR, "tdist_amsmmala_traceplot.pdf"), width=10, height=6)

plot(
  1:npostburnin,
  chains[, pi],
  type="l",
  ylim=c(-4.5, 4.5),
  col="steelblue2",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(-4, 4, by=4),
  labels=seq(-4, 4, by=4),
  cex.axis=1.8,
  las=1
)

lines(
  1:npostburnin,
  rep(chainmean, npostburnin),
  type="l",
  col="orangered1",
  lwd=2
)

dev.off()
