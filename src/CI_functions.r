sandwich_est <- function(X, y, model, est = NULL) {
  # calculate Huber-White sandwich estimator (Algorithm 4, line 1)
  N <- length(y)

  Q <- t(X) %*% X / N

  if (is.null(est)) {
    est <- X %*% coef(model)
  }
  resid <- (y - est)^2

  U <- matrix(0, dim(X)[2], dim(X)[2])
  for (i in 1:N) {
    U <- U + X[i, ] %*% t(X[i, ]) * resid[i]
  }
  U <- U / N
  cov_mat <- solve(Q) %*% U %*% solve(Q) / N
  return(cov_mat)
}


get_se <- function(hw_est, pred.grid) {
  # calculate the standard error for a grid of values

  # Algorithm 4, Line 2
  se <- sqrt(diag(pred.grid %*% hw_est %*% t(pred.grid)))
  return(se)
}