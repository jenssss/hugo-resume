window.onscroll = function() {scrollNavbar()};

function scrollNavbar() {
  if (document.body.scrollTop > 5 || document.documentElement.scrollTop > 5) {
      document.getElementById("navbar-main").classList.add("navbar-light-scrolled");
  } else {
      document.getElementById("navbar-main").classList.remove("navbar-light-scrolled");
  }
}
