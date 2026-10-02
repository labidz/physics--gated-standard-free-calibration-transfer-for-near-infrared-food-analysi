# CT-VPdtw (Zou et al., Anal. Methods 2019, doi:10.1039/c9ay01139k) driven from Python.
# Uses the authors' released R code (github.com/HMzhu/CTVPdtw: savgol, snv, pretreat, plscv, plspredtest,
# RMSEP) and the original VPdtw package (github.com/david-clifford/VPdtw), unchanged.
# One change for an inductive protocol shared by all standard-free methods: the warping reference is built
# from the master calibration spectra and the UNLABELLED target adaptation set Xa (the released code pools
# the slave prediction set itself); target test spectra Xt are then warped onto that same reference.
# The SG derivative order is chosen in {1, 2} by the master cross-validation error (no target labels).
suppressMessages({library(VPdtw); library(Matrix)})
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
for (f in c("savgol.R", "pretreat.R", "plscv.R", "plspredtest.R", "pls1_nipals.R")) source(file.path(here, f))
# snv: same arithmetic as the released snv.R ((x - mean)/sd per row, sd with n-1), vectorised for speed
snv <- function(x) { x <- as.matrix(x); (x - rowMeans(x)) / apply(x, 1, sd) }
find <- function(x) which(if (is.logical(x)) x else x != 0)
repmat <- function(A, m, n) matrix(rep(as.vector(A), times = n), nrow = length(A) * m, ncol = n)

readmat <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  d <- readBin(con, "integer", 2, size = 4)
  matrix(readBin(con, "double", d[1] * d[2], size = 8), nrow = d[1], ncol = d[2], byrow = TRUE)
}
writemat <- function(M, path) {
  M <- as.matrix(M); con <- file(path, "wb"); on.exit(close(con))
  writeBin(as.integer(dim(M)), con, size = 4); writeBin(as.vector(t(M)), con, size = 8)
}

warp_rows <- function(Xd, a, hbmin, span, S, m) {
  out <- matrix(NA, nrow(Xd), m - 20)
  for (i in seq_len(nrow(Xd))) {
    best <- Inf
    for (e in 1:span) for (f in 1:S) {
      b <- Xd[i, ] - hbmin
      r <- VPdtw(reference = a, query = b, penalty = dilation(a, e) / S, maxshift = 80)
      wq <- as.vector(r$warpedQuery)
      ok <- which(is.finite(wq))
      if (length(ok) < 2) next
      # robustness (not in the released code): points left unmatched by the warp (NA) are filled by
      # constant extension/linear interpolation from matched neighbours, so the L1 distance is defined
      wq <- approx(ok, wq[ok], xout = seq_along(wq), rule = 2)$y
      wq <- wq[11:(m - 10)]
      dist <- sum(abs(wq - a[11:(m - 10)]))
      if (is.finite(dist) && dist < best) { best <- dist; out[i, ] <- wq + hbmin }
    }
  }
  out[is.na(out)] <- 0
  out
}

run_job <- function(dir) {
  MX <- readmat(file.path(dir, "Xs.bin")); Y <- readmat(file.path(dir, "ys.bin"))
  XA <- readmat(file.path(dir, "Xa.bin")); XT <- readmat(file.path(dir, "Xt.bin"))
  cfg <- readmat(file.path(dir, "cfg.bin"))          # width, order, span, S, A0
  width <- cfg[1]; order <- cfg[2]; span <- cfg[3]; S <- cfg[4]; A0 <- cfg[5]
  derivs <- if (length(cfg) >= 6 && cfg[6] > 0) cfg[6] else c(1, 2)   # optional fixed derivative order
  best <- NULL
  for (deriv in derivs) {
    D <- savgol(MX, width, order, deriv)$D
    my <- as.matrix(MX %*% D); sa <- as.matrix(XA %*% D); st <- as.matrix(XT %*% D)
    m <- ncol(my); hb <- rbind(sa, my); hbmin <- min(hb); a <- colMeans(hb) - hbmin
    h1 <- snv(warp_rows(my, a, hbmin, span, S, m))
    y11 <- matrix(Y, ncol = 1)
    cv <- plscv(h1, y11, A0, 10)
    if (is.null(best) || min(cv$RMSECV) < best$rmsecv)
      best <- list(deriv = deriv, rmsecv = min(cv$RMSECV), A = cv$Optlv, h1 = h1, y11 = y11, st = st, a = a, hbmin = hbmin, m = m)
  }
  best$A <- max(2, best$A)   # plspredtest in the released code requires A >= 2
  rx <- pretreat(best$h1); ry <- pretreat(best$y11)
  res <- pls1_nipals(rx$mx, ry$mx, best$A)
  Wstar <- res$W %*% solve(t(res$P) %*% res$W); Q <- matrix(res$C, nrow = best$A)
  g1 <- snv(warp_rows(best$st, best$a, best$hbmin, span, S, best$m))
  yp <- plspredtest(Wstar, Q, g1, t(rx$para1), t(rx$para2), t(ry$para1), t(ry$para2), best$A)
  writemat(matrix(as.vector(yp), ncol = 1), file.path(dir, "pred.bin"))
  writemat(matrix(c(best$deriv, best$A, best$rmsecv), nrow = 1), file.path(dir, "info.bin"))
}

jobs <- readLines(commandArgs(TRUE)[1])
for (j in jobs) {
  r <- try(run_job(j), silent = TRUE)
  if (inherits(r, "try-error")) writeLines(as.character(r), file.path(j, "error.txt"))
}
