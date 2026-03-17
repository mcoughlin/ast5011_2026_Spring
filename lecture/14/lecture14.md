# General Relativity and Cosmology – Speaking Notes

*(Covers CFN Ch. 2: the metric tensor, FRW cosmology, cosmological redshift, distance measures, the Friedmann equations, and the cosmological fluid.)*

**Intro / Transition:**
"We're now making the jump from stellar astrophysics to the Universe as a whole. For the past several weeks we've been studying individual stars — their structure, evolution, endpoints. Now we're going to zoom all the way out and ask: what is the large-scale structure and evolution of the entire Universe?

Modern cosmology is built on General Relativity, but the good news is that cosmology is actually one of the simplest applications of GR — because we assume very strong symmetries: the Universe is homogeneous and isotropic on large scales. We don't need to solve the full Einstein field equations in their most general form. We just need the special case where the matter distribution is perfectly smooth and symmetric. That simplification is what makes cosmology tractable."

---

# What Is Cosmology?

"Cosmology is the study of the structure and evolution of the entire Universe — its geometry, its composition, and its history from the Big Bang to the present and beyond.

The foundational insight of General Relativity is that gravity is not a force in the Newtonian sense. Instead, matter and energy curve space-time, and objects move along the straightest possible paths — called geodesics — through that curved space-time. As John Wheeler famously put it: 'Space tells matter how to move; matter tells space how to curve.'

This is fundamentally different from classical physics, where space and time are absolute, rigid, and independent of matter. In GR, the geometry of the Universe is dynamic — it evolves in response to the matter and energy it contains. That's why the expansion of the Universe is possible: the geometry itself is changing.

Now, you might worry that GR is extremely complicated — and in general, it is. The full Einstein field equations are ten coupled, nonlinear partial differential equations. But for cosmology, we exploit the fact that the Universe looks the same everywhere and in every direction on large enough scales. These symmetry assumptions — homogeneity and isotropy — reduce the problem enormously. We'll see that the entire expansion history is governed by just two equations: the Friedmann equations."

---

# The Metric Tensor

"The central mathematical object in GR is the metric tensor $g_{\mu\nu}$. The metric is essentially the generalization of the Pythagorean theorem to curved spaces. It tells you how to measure distances, angles, areas, and volumes — all the geometric information you need.

In flat Euclidean space with Cartesian coordinates, the distance between two nearby points is just $dl^2 = dx^2 + dy^2 + dz^2$. That's the Pythagorean theorem. The metric tensor is just the identity matrix — ones on the diagonal, zeros elsewhere.

But now suppose you switch to polar coordinates. The same flat space has the metric $dl^2 = dr^2 + r^2 d\theta^2$. The metric tensor is no longer the identity — it has an $r^2$ in one of the components. But the underlying space hasn't changed! It's still flat. The key insight is that the components $g_{\mu\nu}$ depend on your coordinate system, but the interval $ds^2$ is invariant — it's the same physical quantity regardless of how you label points.

In special relativity, we extend this to space-time: $ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2$. This is the Minkowski metric. The minus signs encode the fact that time and space are fundamentally different — you can move freely in space, but you always move forward in time.

In GR, the metric becomes more complicated because space-time is curved. The metric components $g_{\mu\nu}$ are no longer constants — they can depend on position and time, and they encode all the information about gravity. Gravitational fields are literally encoded in the curvature of the metric."

---

## 2D Surfaces of Constant Curvature

"Before we tackle the full 4D space-time metric of the Universe, let's build intuition with 2D surfaces — surfaces we can actually visualize.

Consider a 2D sphere of radius $a$ — like the surface of the Earth. If you live on this surface, you can set up coordinates $(\theta, \phi)$ — latitude and longitude. The metric is $dl^2 = a^2(d\theta^2 + \sin^2\theta \, d\phi^2)$. Notice that distances near the poles ($\theta$ near 0 or $\pi$) are compressed in the $\phi$ direction — the circles of constant latitude get smaller. That's curvature in action.

Now we can define a new radial coordinate $r = \sin\theta$. This converts the metric to $dl^2 = a^2(dr^2/(1 - r^2) + r^2 d\phi^2)$. The parameter $a$ sets the overall scale — it's the radius of the sphere. If I double $a$, all distances on the sphere double. So $a$ acts as a scale factor for the geometry.

For a saddle surface — a surface with negative curvature, like a Pringles chip — the metric has the same form but with $1 + r^2$ instead of $1 - r^2$ in the denominator.

We can unify all three cases — sphere, flat plane, and saddle — with a single curvature parameter $K$:
- $K = +1$: positive curvature (sphere)
- $K = 0$: zero curvature (flat plane)
- $K = -1$: negative curvature (saddle)

The unified metric is $dl^2 = a^2(dr^2/(1 - Kr^2) + r^2 d\phi^2)$. This is important because it's exactly the spatial part of the cosmological metric. To get the metric of the Universe, we'll just extend this to 3D and add in the time dimension."

---

## Exercise 1: Expansion and the Scale Factor

"In the first exercise, you'll work with the scale factor $a(t)$ to understand how distances change in an expanding Universe.

The key relationship to internalize is $a = 1/(1+z)$, where we've normalized $a_0 = 1$ today. So at redshift $z = 2$, the scale factor was $a = 1/3$. That means the Universe has expanded by a factor of 3 since the light was emitted. Every distance — between galaxies, between galaxy clusters — was one-third of what it is today.

If a galaxy is currently at a comoving distance of 5000 Mpc, its proper distance at $z = 2$ was $a \times r_{\rm comoving} = (1/3) \times 5000 \approx 1667$ Mpc. The comoving distance is a label that doesn't change with expansion — think of it as coordinates painted on the expanding fabric of space. The proper distance is the actual physical distance at any given time, and it scales with $a$.

When you plot proper distance versus redshift, you'll find it decreases monotonically toward higher $z$. There's no minimum — the proper distance just goes as $l \propto 1/(1+z)$. This makes sense: the Universe was simply smaller in the past. The comoving coordinate of the galaxy hasn't changed — it's the ruler that's shrunk."

---

# The FRW Metric and the Cosmological Principle

"Now we get to the heart of cosmological geometry. The cosmological principle states that the Universe is homogeneous and isotropic on large scales.

What does this actually mean observationally? Homogeneity means that if you could teleport to any random point in the Universe, the average properties — density, temperature, expansion rate — would be the same as here. No place is special. Isotropy means that if you look out in any direction, on average you see the same thing — the same number of galaxies, the same statistical properties.

These are strong claims, and they're supported by observations: the cosmic microwave background is uniform to about one part in 100,000 in all directions; galaxy surveys show that on scales above about 100 Mpc, the distribution of matter is statistically homogeneous.

These symmetry assumptions are incredibly powerful mathematically. Isotropy alone tells you that the only allowed global motion is a uniform expansion or contraction — there can't be a preferred direction of flow. Homogeneity tells you the metric components can't depend on where you are. Together, they uniquely determine the spatial metric up to a single free function — the scale factor $a(t)$ — and a discrete parameter — the curvature $K$.

This gives us the FRW metric: $ds^2 = c^2 dt^2 - a^2(t)[dr^2/(1 - Kr^2) + r^2(d\theta^2 + \sin^2\theta \, d\phi^2)]$. Notice it's just our 2D surface metric extended to 3D, with a time coordinate added. The scale factor $a(t)$ is the one unknown function we need to determine — and the Friedmann equations will tell us what it is."

---

## Proper and Comoving Distance

"Let me be very precise about these two distance concepts because they come up constantly in cosmology, and confusing them is a common source of errors.

The comoving distance $\chi(r)$ is a coordinate label. It's the distance you'd measure if you could somehow freeze the expansion of the Universe at this instant and lay down rulers. It doesn't change with time — galaxies have fixed comoving coordinates (as long as they're not moving relative to the Hubble flow).

The proper distance $l = a(t) \chi(r)$ is the actual physical distance at time $t$. It changes because $a(t)$ changes. Right now, a distant galaxy might be at a proper distance of 5000 Mpc. A billion years from now, the same galaxy — still at the same comoving coordinate — will be at a larger proper distance because $a$ has increased.

An analogy: imagine ants on an inflating balloon. The comoving distance is the angle between two ants on the balloon — it doesn't change as the balloon inflates. The proper distance is the actual arc length between them, which grows as the balloon gets bigger.

For a flat Universe ($K = 0$), the comoving distance is simply $\chi = r$. For curved Universes, there's a correction factor — $\sin^{-1} r$ for closed and $\sinh^{-1} r$ for open geometries."

---

## Conformal Time

"Conformal time $\tau$ is a mathematical convenience that comes up often in theoretical cosmology. It's defined by $d\tau = c\,dt/a(t)$, so $\tau = \int_0^t c\,dt'/a(t')$.

The physical meaning is elegant: conformal time equals the total comoving distance that light could have traveled since the Big Bang. In conformal coordinates, light rays travel at 45-degree angles on a space-time diagram — just like in special relativity. This makes it much easier to think about causal structure: which regions of the Universe can communicate with each other.

We won't use conformal time extensively in this course, but it's good to know what it is because you'll encounter it frequently if you read cosmology papers. The particle horizon — the maximum distance from which light could have reached us since the Big Bang — is simply $\tau_0$, the conformal time today."

---

## The Hubble Parameter

"The Hubble parameter $H(t) = \dot{a}/a$ is the fundamental quantity that describes the expansion rate. It has units of inverse time, though we usually express it in the odd but convenient units of km/s/Mpc.

What does $H$ physically mean? It says that the proper distance between any two fundamental observers changes at a rate $dl/dt = H \times l$. If two galaxies are separated by 1 Mpc, they're moving apart at $H_0 \approx 70$ km/s. If they're separated by 100 Mpc, they're receding at 7000 km/s. The recession velocity is proportional to distance — that's Hubble's law.

The present-day value $H_0$ is called the Hubble constant, though it's really only constant in space — it changes with time as the expansion rate evolves. The inverse $1/H_0 \approx 14$ Gyr gives a rough estimate of the age of the Universe — the Hubble time. The actual age depends on the cosmological model, but the Hubble time sets the right order of magnitude.

You may have heard of the 'Hubble tension' — the fact that different methods of measuring $H_0$ give slightly different answers. CMB-based measurements give about 67 km/s/Mpc, while local distance-ladder measurements give about 73 km/s/Mpc. This is currently one of the most active areas of research in cosmology."

---

## Cosmological Redshift

"Cosmological redshift is one of the most important observational quantities in cosmology, so let's make sure we understand exactly what it means and where it comes from.

A photon traveling through an expanding Universe has its wavelength stretched by the expansion. If the photon was emitted when the scale factor was $a_{\rm em}$ and observed today when $a_0 = 1$, the wavelength ratio is $\lambda_{\rm obs}/\lambda_{\rm em} = a_0/a_{\rm em} = 1 + z$. So the redshift $z$ directly tells us the scale factor at emission: $a = 1/(1+z)$.

A photon from $z = 1$ was emitted when the Universe was half its current size. A photon from $z = 1100$ — the cosmic microwave background — was emitted when the Universe was about 1000 times smaller than today.

I want to emphasize something important: this is NOT a Doppler effect. In a Doppler shift, the source is moving through space, and the frequency changes because of that relative motion. In cosmological redshift, the galaxies aren't moving through space — space itself is expanding, and the wavelength of the photon stretches with it. The metric is changing between the time the photon was emitted and the time it's observed. The distinction matters because cosmological recession velocities can exceed the speed of light — galaxies beyond a certain distance are receding faster than $c$ — and that's perfectly fine in GR. It's not a violation of special relativity because nothing is moving through space faster than light; it's space itself that's expanding."

---

## Peculiar Velocities

"In the real Universe, galaxies don't sit perfectly at rest in the Hubble flow. They have peculiar velocities — motions relative to the cosmological rest frame — caused by gravitational interactions. A galaxy might be falling into a galaxy cluster at several hundred km/s, or it might be part of a group with its own internal motions.

The observed redshift of a galaxy combines both effects: $1 + z_{\rm obs} = (1 + z_{\rm cos})(1 + z_{\rm pec})$. For nearby galaxies, the peculiar velocity contribution can be significant. Our own Local Group is moving at about 600 km/s relative to the CMB rest frame — that's a peculiar velocity. For a galaxy at only 10 Mpc, this would completely dominate over the Hubble flow.

In the non-relativistic limit, the formula simplifies to $z_{\rm obs} \approx z_{\rm cos} + v_{\rm pec}/c \times (1 + z_{\rm cos})$. The $(1 + z_{\rm cos})$ factor accounts for the fact that the peculiar velocity was at the time of emission, not today.

An important theoretical result: peculiar velocities of freely moving particles decay as $v_{\rm pec} \propto a^{-1}$. The expansion of the Universe acts like a friction — it slows things down relative to the comoving frame. This is why the Universe becomes smoother over time (absent gravitational collapse). A particle with 1000 km/s of peculiar velocity at $z = 10$ would have only about 100 km/s today."

---

## Distance Measures

"In an expanding Universe, the concept of 'distance' becomes subtle. There's no single number you can point to and say 'that's the distance.' Different physical measurements give different answers, and all of them are equally valid.

The three main distance measures are:

**Comoving distance** $d_C$: This is the distance you'd measure today if you could freeze the expansion and lay down rulers. It corresponds to the coordinate distance in the FRW metric.

**Angular diameter distance** $d_A = d_C/(1+z)$: This is defined by the relation $\theta = D/d_A$, where $D$ is the physical size of an object and $\theta$ is its angular size on the sky. In Euclidean space, this would just be the regular distance. In an expanding Universe, it's modified because the photons were emitted when the object was closer, so it subtends a larger angle than you'd expect from its current distance.

Here's the really counterintuitive result: $d_A$ reaches a maximum at about $z \approx 1.6$ and then *decreases* at higher redshift. This means that objects beyond $z \approx 1.6$ actually appear *larger* on the sky the further away they are! This is a unique prediction of expanding-Universe cosmology — in a static Euclidean universe, more distant objects always appear smaller.

**Luminosity distance** $d_L = d_C(1+z)$: This is defined by the relation $f = L/(4\pi d_L^2)$, where $L$ is the intrinsic luminosity and $f$ is the observed flux. It grows faster than the comoving distance because of two effects: each photon loses energy by a factor of $(1+z)$ due to redshift, and photons arrive less frequently by another factor of $(1+z)$ due to time dilation.

These distances are related by the Etherington reciprocity relation: $d_L = (1+z)^2 d_A$. This is a fundamental result that holds in any FRW cosmology and is used as a consistency check in observational cosmology."

---

## Exercise 2: Cosmological Redshift and Distances

"In the second exercise, you'll compute the scale factor at various redshifts. The calculation is straightforward — $a = 1/(1+z)$ — but the results are illuminating. At $z = 1100$, the scale factor is about $9 \times 10^{-4}$, meaning the Universe was about 0.09% of its current size when the CMB was emitted. That's an expansion by a factor of 1100 — every distance has grown by more than three orders of magnitude.

You'll also numerically integrate the comoving distance $d_C(z) = (c/H_0) \int_0^z dz'/E(z')$ where $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$. This integral has no closed-form solution for a general $\Lambda$CDM cosmology, so numerical integration is essential. This is one of the bread-and-butter calculations in observational cosmology.

When you plot all three distance measures, the key feature to notice is the turnover in $d_A$ at $z \approx 1.6$. This has practical consequences: it means the angular size of the sound horizon at the surface of last scattering ($z = 1100$) is about 1 degree on the sky — the characteristic angular scale of the CMB acoustic peaks. If the Universe weren't expanding, or if the geometry were different, this angle would change."

---

# The Cosmological Fluid

"Now we need to understand what the Universe is made of and how each component evolves as the Universe expands. We treat the contents of the Universe as a perfect fluid — or really, a mixture of several fluids — characterized by an energy density $\rho$ and a pressure $P$.

The key quantity is the equation of state parameter $w = P/(\rho c^2)$. Each component of the cosmological fluid has a characteristic value of $w$, and this determines how its energy density evolves with the scale factor.

This follows from the first law of thermodynamics applied to a comoving volume. For adiabatic expansion ($dQ = 0$), we get the continuity equation $d\rho/da + 3(\rho + P/c^2)/a = 0$. For constant $w$, this integrates to $\rho \propto a^{-3(1+w)}$. This is one of the most important results in cosmology — it connects the equation of state to the expansion history."

---

## Non-Relativistic Matter

"Baryons and dark matter are non-relativistic — their rest mass energy vastly exceeds their kinetic energy, so $w \approx 0$. This gives $\rho_m \propto a^{-3}$.

The physical interpretation is simple: the number of particles is conserved, and the volume grows as $a^3$, so the number density — and hence the mass-energy density — drops as $a^{-3}$. It's just dilution.

Since $w \approx 0$, the pressure is negligible. Cosmologists call this 'dust' — not because it's actually dust, but because it's pressureless. At the cosmic level, dark matter particles and hydrogen atoms are both effectively pressureless compared to their rest mass energy.

The temperature of non-relativistic matter drops as $T \propto a^{-2}$ from adiabatic cooling. This is steeper than you might expect — it's $a^{-2}$ rather than $a^{-1}$ because the matter is non-relativistic, so the relevant thermodynamic relation is $T \propto v^2 \propto a^{-2}$."

---

## Radiation

"Photons and other relativistic particles have $w = 1/3$, giving $\rho_r \propto a^{-4}$.

Why $a^{-4}$ instead of $a^{-3}$? There are two effects: the number density of photons drops as $a^{-3}$ (dilution), and each photon's energy drops as $a^{-1}$ (redshift). Together, the energy density drops as $a^{-3} \times a^{-1} = a^{-4}$.

The radiation temperature drops as $T \propto a^{-1}$, which follows from $E = h\nu \propto a^{-1}$ and the fact that a thermal spectrum remains thermal under uniform redshifting — the shape of the Planck distribution is preserved, just with a lower temperature.

Because radiation dilutes faster than matter ($a^{-4}$ vs $a^{-3}$), there must have been an epoch in the early Universe when radiation dominated over matter. This crossover happens at $z_{\rm eq} \approx 3400$. Before that, the Universe was radiation-dominated; after that, matter-dominated. The physics of these two eras is quite different — for example, perturbations grow differently in each regime."

---

## Vacuum Energy

"Vacuum energy is the most exotic and counterintuitive component. It has $w = -1$, which means $\rho_\Lambda = {\rm const}$ — the energy density doesn't dilute at all as the Universe expands!

This follows from the first law of thermodynamics. Consider expanding a box of vacuum: the energy increases because you're creating more vacuum, each unit with energy density $\rho_\Lambda$. For energy to be conserved, the pressure must be negative: $P = -\rho_\Lambda c^2$. The negative pressure provides the work needed to create the new vacuum energy.

This is deeply counterintuitive. In everyday experience, pressure is always positive. But in GR, negative pressure has a dramatic consequence: it causes the expansion to accelerate. In the Friedmann acceleration equation, $\ddot{a}/a = -(4\pi G/3)(\rho + 3P/c^2)$, a sufficiently negative pressure — specifically $P < -\rho c^2/3$ — makes $\ddot{a} > 0$. That's accelerated expansion.

The discovery that the expansion of the Universe is actually accelerating — made in 1998 using Type Ia supernovae — was one of the most surprising results in the history of physics and won Perlmutter, Schmidt, and Riess the 2011 Nobel Prize. The simplest explanation is a cosmological constant $\Lambda$, but the theoretical prediction from quantum field theory exceeds the observed value by about 120 orders of magnitude — the notorious 'cosmological constant problem.'"

---

## Energy Density Evolution

"On a log-log plot of energy density versus scale factor, the three components appear as straight lines with different slopes — that's because $\rho \propto a^{-n}$ becomes $\log\rho = -n\log a + {\rm const}$.

Radiation has slope $-4$, matter has slope $-3$, and dark energy has slope $0$ (a horizontal line). Because these slopes are different, different components inevitably dominate at different epochs:

- **Radiation era** ($z > 3400$): Radiation's $a^{-4}$ dominates because at very small $a$, $a^{-4} \gg a^{-3} \gg a^0$.
- **Matter era** ($0.3 < z < 3400$): Matter's $a^{-3}$ dominates. The Universe is decelerating because gravity is slowing the expansion.
- **Dark energy era** ($z < 0.3$): Dark energy's constant density dominates. The expansion is accelerating.

We are currently living in the transition from matter to dark energy domination. Looking forward, dark energy will increasingly dominate, and the expansion will become exponential — every Hubble time, distances double. Eventually, distant galaxies will be carried beyond our observable horizon."

---

# The Friedmann Equations

"Now we come to the dynamical equations that govern the expansion. Plugging the FRW metric into Einstein's field equations — which I won't derive here — gives two independent equations:

The first Friedmann equation: $(\dot{a}/a)^2 = (8\pi G/3)\rho - Kc^2/a^2$. This relates the expansion rate to the total energy density and the curvature. It's essentially an energy conservation equation — the kinetic energy of expansion (left side) equals the gravitational potential energy (first term on right) minus a curvature term.

The acceleration equation: $\ddot{a}/a = -(4\pi G/3)(\rho + 3P/c^2)$. This tells you whether the expansion is speeding up or slowing down. Notice the $\rho + 3P/c^2$ combination — this is why negative pressure (like vacuum energy with $P = -\rho c^2$) can cause acceleration.

In the standard parameterization using density parameters $\Omega_x = \rho_x/\rho_{\rm crit}$:

$$H^2(z) = H_0^2[\Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda + \Omega_K(1+z)^2]$$

This is the master equation of observational cosmology. Given the density parameters and $H_0$, you can compute distances, ages, volumes — everything. The critical density $\rho_{\rm crit} = 3H^2/(8\pi G)$ is the density needed for a flat Universe, and observations show $\Omega_{\rm tot} \approx 1$ — the Universe is indeed very close to flat."

---

## Exercise 3: Age of the Universe

"In the third exercise, you'll integrate the Friedmann equation to compute the age of the Universe. The integral $t_0 = (1/H_0)\int_0^\infty dz/[(1+z)E(z)]$ has no simple closed form for general cosmologies, so again we need numerical integration.

You'll compare three models:

**Einstein-de Sitter** ($\Omega_m = 1$, $\Omega_\Lambda = 0$): This gives $t_0 = (2/3)/H_0 \approx 9.3$ Gyr. This is a problem — it's younger than the oldest globular clusters, which are about 12-13 Gyr old. An Einstein-de Sitter Universe is too young to contain its own stars! This was one of the original motivations for considering a cosmological constant.

**Open** ($\Omega_m = 0.3$, $\Omega_\Lambda = 0$): This gives about 11.3 Gyr — marginal, but still uncomfortably young.

**$\Lambda$CDM** ($\Omega_m = 0.3$, $\Omega_\Lambda = 0.7$): This gives about 13.5 Gyr — comfortably older than the oldest stars. Dark energy makes the Universe older because the expansion was slower in the past than in a decelerating model. Since the Universe took a more leisurely path to its current size, it had more time.

The Hubble time $t_H = 1/H_0 \approx 14$ Gyr sets the scale. In the Einstein-de Sitter model, the age is exactly $2/3$ of the Hubble time. In $\Lambda$CDM, it's about 96% of the Hubble time."

---

## Scale Factor Evolution

"You can also see the difference between cosmologies by plotting $a(t)$ — the scale factor as a function of cosmic time.

In Einstein-de Sitter, $a \propto t^{2/3}$ — the expansion decelerates forever. The curve bends downward, always slowing.

In the open model, the expansion also decelerates but less strongly. Eventually, it approaches free streaming — constant velocity expansion.

In $\Lambda$CDM, the story is richer. At early times, the expansion decelerates just like Einstein-de Sitter — because dark energy is negligible compared to matter at high redshift. But around $z \sim 0.7$ (about 6-7 billion years ago), dark energy starts to dominate, and the expansion begins to accelerate. The curve inflects upward. Looking into the future, the expansion becomes exponential — $a \propto e^{Ht}$ — and the Universe doubles in size every $1/H \approx 14$ Gyr.

This transition from deceleration to acceleration is one of the most important features of our Universe's expansion history. It was directly confirmed by the SN Ia observations in 1998."

---

## Observational Constraints

"Let me briefly summarize how we know the cosmological parameters.

The CMB is our most powerful cosmological probe. The angular position of the first acoustic peak at $\ell \approx 200$ tells us the Universe is flat: $\Omega_{\rm tot} \approx 1$. The relative heights of the peaks encode the baryon density ($\Omega_b \approx 0.044$) and total matter density ($\Omega_m \approx 0.26$). Planck has measured these with exquisite precision.

Type Ia supernovae — standardizable candles — revealed dark energy. By measuring the luminosity distance to supernovae at $z \sim 0.5-1$, Perlmutter, Schmidt, and Riess showed that distant supernovae are about 25% fainter than expected in a decelerating Universe. This means the expansion has been accelerating, requiring $\Omega_\Lambda \approx 0.7$.

The Hubble constant is measured locally using the distance ladder — Cepheids calibrate SN Ia, giving $H_0 \approx 72$ km/s/Mpc — and from the CMB, which gives $H_0 \approx 67$ km/s/Mpc. The discrepancy is the Hubble tension.

The CMB temperature today is $T_0 = 2.725$ K, giving a radiation density of $\Omega_r \approx 8 \times 10^{-5}$ — tiny today, but dominant in the early Universe."

---

## Density Parameter Evolution with Redshift

"An important result that simplifies many calculations: at high redshift, regardless of the present-day cosmological parameters, all cosmologies with matter converge to Einstein-de Sitter behavior. That is, $\Omega_m(z) \to 1$ and $\Omega_\Lambda(z) \to 0$ as $z \to \infty$.

Why? Because matter density grows as $(1+z)^3$ and dark energy is constant. At $z = 10$, matter density is 1000 times its present value while dark energy hasn't changed. So dark energy becomes negligible. Similarly, curvature goes as $(1+z)^2$ — also negligible compared to $(1+z)^3$.

This is a huge simplification for early-Universe calculations. You don't need to know the precise cosmological parameters — at $z > 10$, everything is effectively Einstein-de Sitter. This is why we can study the physics of the CMB ($z = 1100$) or Big Bang nucleosynthesis ($z \sim 10^9$) without worrying about dark energy."

---

## The Linear Growth Factor

"The Friedmann equations don't just tell us about the smooth background Universe — they also govern how density perturbations grow. In the linear regime ($\delta \equiv \delta\rho/\rho \ll 1$), perturbation growth is described by a single function: the linear growth factor $D_+(a)$.

During matter domination, $D_+ \propto a$ — perturbations grow proportionally to the scale factor. If a region is 1% overdense at $z = 100$, it's 10% overdense at $z = 10$, and so on. This is gravitational instability in action — overdense regions attract more matter, becoming denser over time.

During radiation domination, growth is suppressed. This is the Meszaros effect: radiation pressure prevents matter from collapsing on scales smaller than the Jeans length, and the rapid expansion during radiation domination dilutes perturbations faster than gravity can amplify them. Perturbations effectively freeze — they stop growing until matter takes over.

When dark energy starts to dominate at $z \sim 0.7$, growth slows again and eventually freezes out. Dark energy's negative pressure accelerates the expansion, pulling things apart faster than gravity can pull them together.

This growth factor is the direct bridge from the Friedmann equations to the structures we observe today — galaxies, clusters, filaments, and the cosmic web. The amplitude of the growth factor at $z = 0$ compared to the initial conditions at the CMB determines the normalization of the matter power spectrum, parameterized by $\sigma_8 \approx 0.8$."
