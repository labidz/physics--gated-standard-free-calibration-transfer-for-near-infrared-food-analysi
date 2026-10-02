###
### $Id: find.R 29 2022-05-30 23:02:22Z proebuck $
###
### Find indices of nonzero elements.
###


##-----------------------------------------------------------------------------
find <- function(x) {
    expr <- if (is.logical(x)) {
                x
            } else {
                x != 0
            }
    which(expr)
}

