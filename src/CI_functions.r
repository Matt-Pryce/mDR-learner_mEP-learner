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



dr_wboot <- function(X, y, dim, pred.grid, B = 200) {
  # Algorithm 2 & 6, weighted bootstrap for gDR-Learner
  # set.seed(88)

  n <- length(y)
  coef.b <- matrix(0, B, dim + 1)
  for (b in 1:B) {
    v <- rexp(n)
    v <- v / sum(v)
    coef.b[b, ] <- coef(lm(y ~ X - 1, weights = v))
  }

  boot.vals <- pred.grid %*% t(coef.b)
  return(boot.vals)
}

t_boot_se <- function(
  df,
  dim,
  ipsw1,
  ipsw0,
  pred.grid,
  preds,
  subgroup = NULL,
  B = 200
) {
  # Algorithm 3 & 5, bootstrap for gT-Learner standard errors

  df$ipsw <- ifelse(df$A == 1, ipsw1, ipsw0)
  df1 <- df %>% filter(A == 1 & S == 1)
  df0 <- df %>% filter(A == 0 & S == 1)

  n1 <- nrow(df1)
  n0 <- nrow(df0)

  coef.b1 <- matrix(0, B, dim + 1)
  coef.b0 <- matrix(0, B, dim + 1)

  # set.seed(88)
  for (b in 1:B) {
    # bootstrap sample indices for each treatment arm, stratified by subgroup

    sample1 <- df1 %>%
      group_by(across(all_of(subgroup))) %>%
      slice_sample(prop = 1, replace = TRUE) %>%
      dplyr::select(-c("A", "S"))
    sample0 <- df0 %>%
      group_by(across(all_of(subgroup))) %>%
      slice_sample(prop = 1, replace = TRUE) %>%
      dplyr::select(-c("A", "S"))

    X1 <- as.matrix(sample1 %>% dplyr::select(-c("Y", "ipsw")))
    X0 <- as.matrix(sample0 %>% dplyr::select(-c("Y", "ipsw")))

    # Algorithm 3, line 3 option 1
    coef.b1[b, ] <- coef(glm(
      sample1$Y ~ X1 - 1,
      weights = sample1$ipsw,
      family = "binomial"
    ))
    coef.b0[b, ] <- coef(glm(
      sample0$Y ~ X0 - 1,
      weights = sample0$ipsw,
      family = "binomial"
    ))

    while (any(is.na(coef.b1[b, ]) | is.na(coef.b0[b, ]))) {
      # | any(coef.b0[b, ] > 100000) | any(coef.b1[b,] > 100000)
      # bootstrap sample indices for each treatment arm
      print(paste("again", b))
      sample1 <- df1 %>%
        group_by(across(all_of(subgroup))) %>%
        sample_n(n1, replace = TRUE) %>%
        dplyr::select(-c("A", "S"))
      sample0 <- df0 %>%
        group_by(across(all_of(subgroup))) %>%
        sample_n(n0, replace = TRUE) %>%
        dplyr::select(-c("A", "S"))

      X1 <- as.matrix(sample1 %>% dplyr::select(-c("Y", "ipsw")))
      X0 <- as.matrix(sample0 %>% dplyr::select(-c("Y", "ipsw")))

      # Algorithm 3, line 3 option 1
      coef.b1[b, ] <- coef(glm(
        sample1$Y ~ X1 - 1,
        weights = sample1$ipsw,
        family = "binomial"
      ))
      coef.b0[b, ] <- coef(glm(
        sample0$Y ~ X0 - 1,
        weights = sample0$ipsw,
        family = "binomial"
      ))

      print(coef.b1[b, ])
      print(coef.b0[b, ])
    }
  }

  # get fitted values on grid points
  boot.vals1 <- pred.grid %*% t(coef.b1)
  boot.vals0 <- pred.grid %*% t(coef.b0)

  boot.vals <- logit2prob(boot.vals1) - logit2prob(boot.vals0)

  # calculate standard errors for each observation over the B bootstrap replicates
  cate_se <- apply(boot.vals, 1, sd)
  # print(head(boot.vals1[,94]))
  # print(head(boot.vals0[,94]))
  return(list(boot.vals = boot.vals, cate_se = cate_se))
}


undersmooth_hal <- function(
  X,
  Y,
  fit_init,
  Nlam = 20,
  family = c("gaussian", "binomial", "poisson", "cox"),
  weights = NULL
) {
  n = length(Y)

  nonzero_col_init = which(fit_init$coefs[-1] != 0)
  if (length(nonzero_col_init) == 0) {
    res <- list(
      "lambda_init" = fit_init$lambda_star,
      "lambda_under" = fit_init$lambda_star
    )
    return(res)
  }

  #  refit on a grid of lambda (scaled on the lambda from cv-hal)
  us_lambda <- fit_init$lambda_star * 10^seq(from = 0, to = -3, length = Nlam)
  us_fit <- glmnet(
    fit_init$x_basis,
    Y,
    lambda = us_lambda,
    family = family,
    standardize = FALSE,
    weights = weights
  )

  if (identical(us_fit$df, 0)) {
    res <- list(
      "lambda_init" = fit_init$lambda_star,
      "lambda_under" = fit_init$lambda_star
    )
    return(res)
  }

  preds_init <- predict(fit_init, new_data = X)
  resid_init <- preds_init - Y

  if (family != "binomial") {
    pred_mat <- predict(us_fit, fit_init$x_basis)
  } else {
    pred_mat <- predict(us_fit, fit_init$x_basis, type = "response")
  }
  resid_mat <- pred_mat - Y

  ## estimates of sd in each direction using initial fit
  basis_mat_init <- as.matrix(fit_init$x_basis)
  basis_mat_init <- as.matrix(basis_mat_init[, nonzero_col_init])

  sd_est <- apply(basis_mat_init, 2, function(phi) sd(resid_init * phi))

  ## calculate scores
  max_score <- get_maxscore(
    basis_mat = basis_mat_init,
    resid_mat = resid_mat,
    sd_est = sd_est,
    Nlam = Nlam,
    us_fit = us_fit
  )

  ## get the first lambda that satisfies the criteria
  lambda_under <- us_lambda[max_score <= 1 / (sqrt(n) * log(n))][1] # over under-smoothing

  if (is.na(lambda_under)) {
    res <- list(
      "lambda_init" = fit_init$lambda_star,
      "lambda_under" = fit_init$lambda_star
    )
    return(res)
  }

  # collect results
  coef_mat <- as.matrix(us_fit$beta)

  spec_under <- list("lambda" = us_lambda, "l1_norm" = NA, "n_coef" = NA)

  spec_under$l1_norm <- apply(coef_mat, 2, function(x) {
    sum(abs(x))
  })
  spec_under$n_coef <- apply(coef_mat, 2, function(x) {
    sum(x != 0)
  })

  res <- list(
    "lambda_init" = fit_init$lambda_star,
    "lambda_under" = lambda_under,
    "spec_under" = spec_under
  )
  return(res)
}


IC_based_se <- function(
  X,
  Y,
  hal_fit,
  eval_points,
  family = "binomial",
  X_unpenalized = NULL,
  subset = NULL,
  weights = NULL
) {
  n = length(Y)
  # get coefficients from CV-HAL
  coef <- hal_fit$coefs
  basis_mat <- cbind(1, as.matrix(hal_fit$x_basis))
  nonzero_idx <- which(coef != 0)

  if (length(nonzero_idx) > 0) {
    # select basis functions corresponding to non-zero coefficients
    coef_nonzero <- coef[nonzero_idx]
    basis_mat_nonzero <- as.matrix(basis_mat[, nonzero_idx])

    # get estimates on data (i.e. project X onto basis functions with non-zero coefficients)
    if (family == "binomial") {
      if (is.null(X_unpenalized)) {
        Y_hat = predict(hal_fit, new_data = X, type = "response")
      } else {
        Y_hat = predict(
          hal_fit,
          new_data = X,
          new_X_unpenalized = as.matrix(X_unpenalized),
          type = "response"
        )
      }
    } else {
      if (is.null(X_unpenalized)) {
        Y_hat = predict(hal_fit, new_data = X)
      } else {
        Y_hat = predict(
          hal_fit,
          new_X_unpenalized = as.matrix(X_unpenalized),
          new_data = X
        )
      }
    }

    # subsetting basis functions onto trial only individuals (returns basis matrix of size n_s=1 x number of splines)
    if (!is.null(subset)) {
      basis_mat_nonzero <- basis_mat_nonzero[subset, ]
    }

    # Step 5 part 1, calculate influence curve
    IC_beta <- cal_IC_for_beta(
      X = basis_mat_nonzero,
      Y = Y,
      Y_hat = Y_hat,
      beta_n = coef_nonzero,
      family = family
    )

    se <- c()
    ic_ey <- c()

    if (any(!is.na(IC_beta))) {
      # Precompute basis for all eval points at once — single-row make_design_matrix
      # is unreliable and triggers subscript-out-of-bounds warnings in hal9001
      eval_mat <- as.matrix(eval_points)
      x_basis_all <- hal9001::make_design_matrix(
        eval_mat,
        hal_fit$basis_list,
        p_reserve = 0.75
      )
      x_basis_all_nonzero <- as.matrix(
        cbind(1, x_basis_all)[, nonzero_idx, drop = FALSE]
      )

      for (i in 1:NROW(eval_points)) {
        if (is.null(X_unpenalized)) {
          x_basis_a_nonzero <- x_basis_all_nonzero[i, , drop = FALSE]
        } else {
          X_new <- if (is.vector(eval_points)) {
            as.matrix(eval_points[i])
          } else {
            as.matrix(eval_points[i, , drop = FALSE])
          }
          x_basis_a <- hal9001::make_design_matrix(
            X_new,
            hal_fit$basis_list,
            p_reserve = 0.75
          )
          x_basis_a_nonzero <- as.matrix(cbind(
            1,
            x_basis_a,
            as.matrix(X_unpenalized)
          )[, nonzero_idx])
        }

        IC_EY <- cal_IC_for_EY(
          X_new = x_basis_a_nonzero,
          beta_n = coef_nonzero,
          IC_beta = IC_beta,
          family = family
        )

        ic_ey <- rbind(ic_ey, mean(IC_EY))

        # Step 6, empirical SE
        se[i] <- sqrt(var(IC_EY) / n)
      }
    } else {
      se <- NA
    }
  } else {
    se <- NA
  }

  #return se and ic_ey
  return(list(se = se, IC_EY = ic_ey))
}


cal_IC_for_EY <- function(X_new, beta_n, IC_beta, family = 'binomial') {
  if (!is.matrix(X_new)) {
    X_new <- as.matrix(X_new)
  }

  if (family == 'binomial') {
    d_phi_scaler_new <- as.vector(
      exp(-beta_n %*% t(X_new)) / ((1 + exp(-beta_n %*% t(X_new)))^2)
    )
    d_phi_new <- sweep(X_new, 1, d_phi_scaler_new, `*`)
  } else {
    d_phi_new = X_new
  }

  IC = as.vector(d_phi_new %*% IC_beta)
  return(IC)
}


# calculating efficient influence curves

cal_IC_for_beta <- function(
  X,
  Y,
  Y_hat,
  beta_n,
  family = 'binomial',
  return_tmat = FALSE
) {
  n <- dim(X)[1]
  p <- length(beta_n)
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }

  # 1. calculate score: X'(Y - phi(X))
  res <- Y - Y_hat
  score <- sweep(t(X), 2, res, `*`)

  # 2. calculate the derivative of phi:
  if (family == 'binomial') {
    d_phi_scaler <- as.vector(
      exp(-beta_n %*% t(X)) / ((1 + exp(-beta_n %*% t(X)))^2)
    )
    d_phi <- sweep(X, 1, d_phi_scaler, `*`)
  } else {
    d_phi = -X
  }

  # 3. -E_{P_n}(X d_phi)^(-1)
  tmat <- t(X) %*% d_phi / n
  if (!is.matrix(try(solve(tmat), silent = TRUE))) {
    return(NA)
  }
  tmat <- -solve(tmat)

  # 4. calculate influence curves
  IC <- tmat %*% score

  if (return_tmat) {
    return(list(IC = IC, tmat = tmat))
  }
  return(IC)
}